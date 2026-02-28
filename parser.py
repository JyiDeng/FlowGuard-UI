import re
from collections import namedtuple

# ==========================
# 1) Tokenizer
# ==========================
Token = namedtuple("Token", "type value")

PLANTUML = "@startuml\nstart\n:用户报案;\n\nif (事故是否在保单范围内?) then (Yes)\n    :系统生成理赔编号;\nelse (No)\n    :拒绝理赔并通知用户;\n    stop\nendif\n\n:系统审核资料;\n\nrepeat :检查资料完整性;\nrepeat while (资料不完整?) is (Yes)\n    :通知用户补充资料;\n    :用户提交资料;\n->No;\n\n:审核完成并生成方案;\n\nif (方案是否需人工审核?) then (Yes)\n    :转交人工审核;\nelse (No)\n    :系统自动批准方案;\nendif\n\n:系统发起理赔付款;\n\nif (付款是否成功?) then (Yes)\n    :通知用户付款完成;\n    stop\nelse (No)\n    :通知用户付款失败，重试;\n    stop\nendif\n@enduml"

IF_RE   = re.compile(r"^if\s*\((.*?)\)\s*then\s*\((.*?)\)\s*$")
ELSE_RE = re.compile(r"^else(?:\s*\((.*?)\))?\s*$")
ACT_RE  = re.compile(r"^:(.*);\s*$")
REPEAT_INLINE_RE = re.compile(r"^repeat\s*:(.*);\s*$")
REPEAT_WHILE_RE = re.compile(
    r"^repeat\s+while\s*\((.*?)\)"
    r"(?:\s*is\s*\((.*?)\))?"
    r"(?:\s*not\s*\((.*?)\))?\s*$"
)


def tokenize(text: str):
    tokens = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('@'):
            continue
        if line == 'start':
            tokens.append(Token('START', None))
            continue
        if line == 'end':
            tokens.append(Token('STOP', None))
            continue
        if line == 'stop':
            tokens.append(Token('STOP', None))
            continue
        if line == 'endif':
            tokens.append(Token('ENDIF', None))
            continue
        if line == 'repeat':
            tokens.append(Token('REPEAT', None))
            continue
        m = REPEAT_INLINE_RE.match(line)
        if m:
            label = m.group(1).replace('\\n', ' ').strip()
            tokens.append(Token('REPEAT', label))
            continue
        m = REPEAT_WHILE_RE.match(line)
        if m:
            cond = (m.group(1) or '').strip()
            tlabel = (m.group(2) or '').strip()
            flabel = (m.group(3) or '').strip()
            tokens.append(Token('REPEAT_WHILE', (cond, tlabel, flabel)))
            continue
        m = IF_RE.match(line)
        if m:
            cond, tlabel = m.group(1), m.group(2)
            tokens.append(Token('IF', (cond, tlabel)))
            continue
        m = ELSE_RE.match(line)
        if m:
            flabel = m.group(1) or ''
            tokens.append(Token('ELSE', flabel))
            continue
        m = ACT_RE.match(line)
        if m:
            label = m.group(1).replace('\\n', ' ').strip()
            tokens.append(Token('ACTION', label))
            continue
    return tokens

# ==========================
# 2) AST 节点
# ==========================
class Start: pass
class Stop: pass
class Action:
    def __init__(self, label): self.label = label
class Decision:
    def __init__(self, cond, tlabel, tblock, flabel, fblock):
        self.cond = cond
        self.tlabel = tlabel
        self.tblock = tblock
        self.flabel = flabel
        self.fblock = fblock
class Repeat:
    def __init__(self, block, cond, tlabel='', flabel=''):
        self.block = block
        self.cond = cond
        self.tlabel = tlabel
        self.flabel = flabel

# ==========================
# 3) Parser
# ==========================
def parse(tokens):
    i = 0
    def parse_seq(i):
        stmts = []
        while i < len(tokens):
            tok = tokens[i]
            t = tok.type
            if t in ('ELSE', 'ENDIF', 'REPEAT_WHILE'):
                break
            if t == 'START':
                stmts.append(Start()); i += 1; continue
            if t == 'STOP':
                stmts.append(Stop()); i += 1; continue
            if t == 'ACTION':
                stmts.append(Action(tok.value)); i += 1; continue
            if t == 'IF':
                cond, tlabel = tok.value
                i += 1
                true_block, i = parse_seq(i)
                flabel = ''
                false_block = []
                if i < len(tokens) and tokens[i].type == 'ELSE':
                    flabel = tokens[i].value or ''
                    i += 1
                    false_block, i = parse_seq(i)
                if i >= len(tokens) or tokens[i].type != 'ENDIF':
                    raise SyntaxError('缺少 ENDIF 对应的 if: ' + cond)
                i += 1
                stmts.append(Decision(cond, tlabel, true_block, flabel, false_block))
                continue
            if t == 'REPEAT':
                inline_action = tok.value
                i += 1
                loop_block, i = parse_seq(i)
                if i >= len(tokens) or tokens[i].type != 'REPEAT_WHILE':
                    raise SyntaxError('缺少 REPEAT WHILE 对应的 repeat')
                cond, tlabel, flabel = tokens[i].value
                i += 1
                if inline_action:
                    loop_block = [Action(inline_action)] + loop_block
                stmts.append(Repeat(loop_block, cond, tlabel, flabel))
                continue
            break
        return stmts, i

    ast, _ = parse_seq(i)
    return ast

# ==========================
# 4) Build Graph
# ==========================
class GraphBuilder:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._nid = 0
    def new(self, ntype, label=None):
        nid = f'N{self._nid}'; self._nid += 1
        self.nodes[nid] = {'type': ntype, 'label': label}
        return nid
    def connect(self, sources, target, label=None):
        for s in sources or []:
            if self.nodes[s]['type']!='stop':
                self.edges.append({'from': s, 'to': target, 'label': label})

    def build_seq(self, seq, incoming=None, entry_edge_label=None):
        exits = incoming or []
        first = True
        for stmt in seq:
            if isinstance(stmt, Start):
                nid = self.new('start', 'start')
                self.connect(exits, nid, entry_edge_label if first else None)
                exits = [nid]
            elif isinstance(stmt, Stop):
                nid = self.new('stop', 'stop')
                self.connect(exits, nid, entry_edge_label if first else None)
                exits = [nid]
            elif isinstance(stmt, Action):
                nid = self.new('action', stmt.label)
                self.connect(exits, nid, entry_edge_label if first else None)
                exits = [nid]
            elif isinstance(stmt, Decision):
                d = self.new('decision', stmt.cond)
                self.connect(exits, d, entry_edge_label if first else None)
                t_exits = self.build_seq(stmt.tblock, incoming=[d], entry_edge_label=stmt.tlabel)
                if stmt.fblock:
                    f_exits = self.build_seq(stmt.fblock, incoming=[d], entry_edge_label=stmt.flabel or None)
                else:
                    f_exits = [d]
                # 不再添加 merge 节点，直接将分支出口返回
                exits = t_exits + f_exits
            elif isinstance(stmt, Repeat):
                # PlantUML repeat...repeat while 是后测试循环：先执行循环体，再判断条件。
                loop_entry = self.new('noop', 'repeat')
                self.connect(exits, loop_entry, entry_edge_label if first else None)

                body_incoming = [loop_entry]
                if stmt.block:
                    body_exits = self.build_seq(stmt.block, incoming=body_incoming)
                else:
                    body_exits = body_incoming

                d = self.new('decision', stmt.cond or 'repeat while')
                self.connect(body_exits, d)

                loop_label = stmt.tlabel or 'Yes'
                self.connect([d], loop_entry, loop_label)
                # false/not 分支流向后续语句；当前构图器不携带“待连接边标签”，因此这里只返回决策点。
                exits = [d]
            first = False
        return exits


def build_graph(ast):
    gb = GraphBuilder()
    gb.build_seq(ast)
    return {'nodes': gb.nodes, 'edges': gb.edges}

# ==========================
# 5) Render with networkx
# ==========================
def render_graph(graph):
    import networkx as nx
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = 'SimHei'
    G = nx.DiGraph()
    for nid, info in graph['nodes'].items():
        label = info['label'] if info['label'] else info['type']
        G.add_node(nid, label=label, type=info['type'])
    for e in graph['edges']:
        G.add_edge(e['from'], e['to'], label=e['label'] or '')

    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=2000,
            node_color='lightblue', font_size=10, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.show()

def run(PLANTUML=PLANTUML):
    toks = tokenize(PLANTUML)
    ast = parse(toks)
    graph = build_graph(ast)
    # render_graph(graph)
    return graph

# ==========================
# 6) Run
# ==========================
if __name__ == '__main__':
    toks = tokenize(PLANTUML)
    ast = parse(toks)
    graph = build_graph(ast)
    render_graph(graph)
