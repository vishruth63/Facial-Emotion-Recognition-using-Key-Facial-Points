import os
import matplotlib.pyplot as plt
import networkx as nx

path = 'plots/'

def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5 ):
    def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None, parsed = []):
        if(root not in parsed):
            parsed.append(root)
            if pos == None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            neighbors = G.neighbors(root)

            if parent != None:
                neighbors.remove(parent)
            if len(neighbors)!=0:
                dx = width/len(neighbors)
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    pos = h_recur(G,neighbor, width = dx, vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos,parent = root, parsed = parsed)
        return pos

    return h_recur(G, root, width=2, vert_gap = 0.1, vert_loc = 0, xcenter = 0.5)

def hierarchy_pos_large(G, root, levels=None, width=1., height=1.):
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """
        Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        if parent is not None:
            neighbors.remove(parent)
        for neighbor in neighbors:
            levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        if parent is not None:
            neighbors.remove(parent)
        for neighbor in neighbors:
            pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})

def visualize_tree(tree_edges, file_index, labels, root_node = 0, emotion = "default_emotion"):
    def file_name(is_overlap):
        return path + str(file_index) + '-' + emotion + ('' if is_overlap else '-nonoverlap') + '.pdf'

    G = nx.Graph()
    G.add_edges_from(tree_edges)

    font_size = 10
    node_size = 500
    node_color = "palegreen"

    if not os.path.exists(path):
        os.makedirs(path)

    pos_half = hierarchy_pos(G, root_node)
    plt.figure(figsize=(40, 12))
    plt.title(emotion+ '-' + str(file_index))
    nx.draw(G, pos=pos_half, with_labels=True, node_size=node_size, font_size=font_size, node_color=node_color, labels=labels)
    plt.savefig(file_name(True), dpi=1000)
    plt.clf()

    pos_all = hierarchy_pos_large(G, root_node)
    plt.figure(figsize=(40, 40))
    plt.title(emotion+ '-' + str(file_index))
    nx.draw(G, pos=pos_all, with_labels=True, node_size=node_size, font_size=font_size, node_color=node_color, labels=labels)
    plt.savefig(file_name(False), dpi=1000)
    plt.clf()