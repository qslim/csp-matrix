def relation(num, p1, p2):
    rel = []
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            if p1 - p2 == i - j or p1 - p2 == j - i:
                continue
            rel.append([i, j])
    return rel


