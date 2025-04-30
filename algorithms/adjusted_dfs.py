def adjusted_dfs(start, target, edge_heap, graph):
    stack = [(start, [])]
    seen = set()
    while stack:
        node, path = stack.pop()
        if node == target:
            return path
        seen.add(node)
        outgoing = sorted(graph[node], key=lambda e: -edge_heap.get(e, 0))
        for edge in outgoing:
            _, neighbor = edge
            if neighbor not in seen:
                stack.append((neighbor, path + [edge]))
    return []