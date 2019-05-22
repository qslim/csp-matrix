import xml.etree.ElementTree as ET
import numpy as np


def parser(path):
    print(path)
    tree = ET.parse(path)
    instance = tree.getroot()
    # print(instance.tag, ":", instance.attrib)

    # build vars_map
    domains = instance.find("domains")
    max_dom = 0
    for domain in domains:
        # print(domain.get("nbValues"))
        max_dom = max(max_dom, int(domain.get("nbValues")))

    variables = instance.find("variables")
    num_vars = int(variables.get("nbVariables"))
    vars_map = []

    for variable in variables:
        line = []
        for i in range(max_dom):
            line.append(1)

        vars_map.append(line)

    # mask = np.ones([15, 2])
    # vars_map = np.matmul(vars_map, mask)
    # vars_map = np.where(vars_map < 444, -1, vars_map)
    vars_map = np.array(vars_map)
    # print(vars_map)
    vars_map = np.expand_dims(vars_map, 2)

    # print(vars_map.shape)

    # build rels_map
    relations = instance.find("relations")
    num_rels = int(relations.get("nbRelations"))
    rels_map = []
    rels_map_r = []
    for relation in relations:
        # print("nbTuples: ", relation.get("nbTuples"))
        # print("semantics: ", relation.get("semantics"))
        rel_tst = relation.text.strip()
        # print(rel_tst)
        tuple_list = str(rel_tst).split('|')

        if relation.get("semantics") == "conflicts":
            rel_map = [[1 for _ in range(max_dom)] for _ in range(max_dom)]
            rel_map_r = [[1 for _ in range(max_dom)] for _ in range(max_dom)]
            for t in tuple_list:
                t1 = int(t.split(' ')[0])
                t2 = int(t.split(' ')[1])
                # print(t1 * t2)
                rel_map[t1][t2] = 0
                rel_map_r[t2][t1] = 0
        else:
            rel_map = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
            rel_map_r = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
            for t in tuple_list:
                t1 = int(t.split(' ')[0])
                t2 = int(t.split(' ')[1])
                # print(t1 * t2)
                rel_map[t1][t2] = 1
                rel_map_r[t2][t1] = 1

        # print(rel_map)
        rels_map.append(rel_map)
        rels_map_r.append(rel_map_r)

    for rel_map_r_ in rels_map_r:
        rels_map.append(rel_map_r_)

    # print(np.array(rels_map).shape)
    # print(np.array(rels_map[0]))

    # build cons_map
    constraints = instance.find("constraints")
    num_cons = int(constraints.get("nbConstraints"))
    cons_to_rel = [[-1 for _ in range(num_cons)] for _ in range(num_cons)]

    for constraint in constraints:
        scope = constraint.get("scope")
        c1 = int(str(scope).split(' ')[0][1:])
        c2 = int(str(scope).split(' ')[1][1:])
        reference = int(str(constraint.get("reference"))[1:])
        # print(c1, "and", c2, "is", reference)
        cons_to_rel[c1][c2] = reference
        cons_to_rel[c2][c1] = reference + num_rels

    # print(cons_to_rel)

    cons_map = []
    for i in range(num_vars):
        cube = []
        for j in range(num_vars):
            if i == j:
                cube.append(np.identity(max_dom).tolist())
            elif cons_to_rel[i][j] == -1:
                cube.append([[1 for _ in range(max_dom)] for _ in range(max_dom)])
            else:
                cube.append(rels_map[cons_to_rel[i][j]])
                # cube.append(copy.deepcopy(rels_map[cons_to_rel[i][i]]))

        cons_map.append(cube)

    cons_map = np.array(cons_map)

    # print(cons_map.shape)
    # print(cons_map[0][3])
    # print(cons_map[3][0])

    # return num_vars, max_dom, vars_map, cons_map
    return num_vars, max_dom, vars_map.transpose((0, 2, 1)), cons_map


# parser("/home/ymq/Downloads/rand-2-30-15/rand-2-30-15-306-230-10_ext.xml")
# parser("/home/ymq/csp_benchmark/run_dir/rand-2-30-15-fcd/rand-2-30-15-306-230-fcd-17_ext.xml")

