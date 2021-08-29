from groundtruth.interactive.gh_scanning import get_heightmap, gray2height, height2ascii, scan


def get_height_grid():
    s = scan()
    hm_feature = get_heightmap(s)
    height_b = hm_feature.channel_split()[0]
    height = gray2height(height_b)
    path = height2ascii(height)
    return path


if __name__ == '__main__':
    get_height_grid()
