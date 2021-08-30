import groundtruth.toolbox.compas_utils as cu
import compas.geometry as cg
import compas.datastructures as cd
import datetime
import json


def export_json(dir, data, iter):
    id_num = str(iter).zfill(3)
    filepath = dir + '/{}_toolpaths.json'.format(id_num)
    with open(filepath, 'w') as o:
        json.dump(data, o, indent=4)


def import_json(dir):
    # filepath = dir +
    # with open(filepath, 'r') as i:
    #     data = json.load()
    pass


def flip_y_value(toolpaths):
    for tp in toolpaths:
        for p in tp.points:
            p.y *= (-1)
    return toolpaths


if __name__ == '__main__':
    pass
