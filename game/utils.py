from game.enums import PlayerId, Resource

def DFS(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths


def check_resource_conservation(env, res_tot=39):
    conserved = True
    for res in [Resource.Wood, Resource.Brick, Resource.Wheat, Resource.Sheep, Resource.Ore]:
        sum = env.game.resource_bank[res]
        for player in [PlayerId.Blue, PlayerId.Red, PlayerId.White, PlayerId.Orange]:
            sum += env.game.players[player].resources[res]
        if sum != res_tot:
            conserved = False
            break
    return conserved