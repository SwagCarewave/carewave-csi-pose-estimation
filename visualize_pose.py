import matplotlib.pyplot as plt

POSE_CONNECTIONS = [
    (11,12),
    (11,13),
    (13,15),
    (12,14),
    (14,16),
    (11,23),
    (12,24),
    (23,24),
    (23,25),
    (25,27),
    (24,26),
    (26,28),
]

def draw_pose(points):

    plt.figure(figsize=(5,6))

    for a,b in POSE_CONNECTIONS:
        plt.plot(
            [points[a,0], points[b,0]],
            [points[a,1], points[b,1]]
        )

    plt.scatter(points[:,0], points[:,1])

    plt.gca().invert_yaxis()

    plt.xlim(0,1)
    plt.ylim(1,0)

    plt.grid(True)
    plt.show()