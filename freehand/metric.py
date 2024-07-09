import numpy as np


def frame_volume_overlap(ps_true, ps_pred, precision=0.5):

    """
    :param both arguments are points sets in numpy array
        shape = [xyz-coordinates, four-conner-points, number-of-frames]
    :return the Dice similarity coefficient overlap measure 
    """

    # first create an overall grid
    ps = np.concatenate([ps_true.reshape(3,-1),ps_pred.reshape(3,-1)],1)
    min_max = [np.amin(ps,axis=1),np.amax(ps,axis=1)]    
    grid_xyz = [np.arange(min_max[0][d]-precision/2,min_max[1][d]+precision/2,precision) for d in range(3)]
    gx, gy, gz = np.meshgrid(grid_xyz[0],grid_xyz[1],grid_xyz[2], indexing='xy')
    px = np.stack((gx.reshape((-1,)),gy.reshape((-1,)),gz.reshape((-1,))),axis=0)

    # swap the point order such that they are in (anti-)clockwise orders
    ps_true = ps_true[:,[0,1,3,2],:]
    ps_pred = ps_pred[:,[0,1,3,2],:]
    
    # iterate both point sets to test if grid inside each hexahedron formed by subsequent two frames
    mask_true = iterate_frames(ps_true,px)
    mask_pred = iterate_frames(ps_pred,px)

    # now compute the Dice
    return 2*np.sum(mask_true & mask_pred) / (np.sum(mask_true) + np.sum(mask_pred))


## utility functions
def iterate_frames(pts, px):
    mask = np.full((px.shape[1],), False)
    for idx in range(pts.shape[2]-1):
        ps_hex = np.concatenate((pts[...,idx],pts[...,idx+1]), axis=1) 
        mask = mask | test_inside_hexahedron(px, ps_hex)
        

        # debug:
        # is_inside = test_inside_hexahedron(px, ps_hex)
        # print(np.count_nonzero(is_inside))
        # px_inside = px[np.ix_([True,True,True],is_inside)]
        #
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(ps_hex[0,:], ps_hex[1,:], ps_hex[2,:], marker='x')
        # ax.scatter(px_inside[0,:], px_inside[1,:], px_inside[2,:], marker='.')
        # plt.show()
        #
        # print(idx)


    return mask



def test_inside_hexahedron(px, ps_hex, three_vert=True):

    '''
    Algorithm described in https://stackoverflow.com/questions/72170319/check-if-a-point-is-inside-an-arbitrary-hexahedron
    This is faster than tessellation-based methods, and can be improved by pre-screening bounding box or sphere 
    :px 3-by-n point vectors to test
    :ps 3-by-8 point vectors representing hexahedron conners 
        node 0-3 and node 4-7 are in the same (anti-)clockwise order
    
                  t
                  |
         4--------|-------------7
        /|        |            /|
       / |        |           / |
      /  |        |          /  |
     /   |        |         /   |
    /    |        |        /    |
   /     |        |       /     |
  5----------------------6      |
  |      |        |      |      |
  |      |        o------|---------s
  |      |       /       |      |
  |      0------/--------|------3
  |     /      /         |     /
  |    /      /          |    /
  |   /      /           |   /
  |  /      /            |  /
  | /      r             | /
  |/                     |/
  1----------------------2

    '''

    # compute the normal vectors 
    if three_vert:  # normal vector by three vertices 
        r_pos_nv = np.cross(ps_hex[:,2]-ps_hex[:,1], ps_hex[:,5]-ps_hex[:,1])
        r_neg_nv = np.cross(ps_hex[:,4]-ps_hex[:,0], ps_hex[:,3]-ps_hex[:,0])
        s_pos_nv = np.cross(ps_hex[:,3]-ps_hex[:,2], ps_hex[:,6]-ps_hex[:,2])
        s_neg_nv = np.cross(ps_hex[:,5]-ps_hex[:,1], ps_hex[:,0]-ps_hex[:,1])
        t_pos_nv = np.cross(ps_hex[:,6]-ps_hex[:,5], ps_hex[:,4]-ps_hex[:,5])
        t_neg_nv = np.cross(ps_hex[:,0]-ps_hex[:,1], ps_hex[:,2]-ps_hex[:,1])
    else:  # normal vector by four vertices 
        r_pos_nv = np.cross(ps_hex[:,6]-ps_hex[:,1], ps_hex[:,5]-ps_hex[:,2])
        r_neg_nv = np.cross(ps_hex[:,7]-ps_hex[:,0], ps_hex[:,4]-ps_hex[:,3])
        s_pos_nv = np.cross(ps_hex[:,7]-ps_hex[:,2], ps_hex[:,6]-ps_hex[:,3])
        s_neg_nv = np.cross(ps_hex[:,4]-ps_hex[:,1], ps_hex[:,5]-ps_hex[:,0])
        t_pos_nv = np.cross(ps_hex[:,7]-ps_hex[:,5], ps_hex[:,4]-ps_hex[:,6])
        t_neg_nv = np.cross(ps_hex[:,3]-ps_hex[:,1], ps_hex[:,0]-ps_hex[:,2])
    
    # for opposite nodes 0 (negative) and 6 (positive):
    p0x = (px-ps_hex[:,[0]]).transpose()
    p6x = (px-ps_hex[:,[6]]).transpose()
    is_inside = np.all(np.stack([
        np.inner(p0x,r_neg_nv)<0, 
        np.inner(p0x,s_neg_nv)<0, 
        np.inner(p0x,t_neg_nv)<0, 
        np.inner(p6x,r_pos_nv)<0, 
        np.inner(p6x,s_pos_nv)<0, 
        np.inner(p6x,t_pos_nv)<0],axis=1), axis=1)
    # np.count_nonzero(is_inside)
    return is_inside
