def npathloss(gt, pred, predecessor, num_paths=10, soma=None, debug=False):
    gt_clone = gt.detach().clone().cpu().numpy()
    pred_clone = pred.detach().clone().cpu().numpy()
    predecessor_clone = predecessor.detach().clone().cpu().numpy()
    soma_clone = soma.detach().clone().cpu().numpy()
    soma_clone = tuple([int(soma_clone[0]), int(soma_clone[1]), int(soma_clone[2])])

    bin_pred = (pred_clone > 0.5).astype(int)
    # bin_pred = binary_erosion(bin_pred, iterations=3)
    soma_cc = soma_cc_cc3d(bin_pred, soma_clone)
    soma_cc = binary_dilation(soma_cc, iterations=3)

    non_soma_cc = gt_clone - soma_cc
    non_soma_cc = np.where(non_soma_cc > 0.5, 1, 0).astype(int)

    rand_points = random_foreground_points(non_soma_cc, num_paths)
    pt_loss = 0
    if(debug):
        paths = []
    for start_point in rand_points:
        # print(f"point: {point}, {point[0]}")
        # print(f"type of point: {type(point)}, {type(point[0])}")
        path = find_path_to_source(predecessor_clone, start_point, soma_clone)
        if(debug):
            paths.append(path)
        # 反转path
        path = path[::-1]
        mask_path = np.zeros_like(gt_clone)
        mask_path_from_soma = np.zeros_like(gt_clone)
        continue_from_soma_flag = True
        for point in path:
            mask_path[point] = 1
            if(bin_pred[point] < 0.5):
                continue_from_soma_flag = False
            if(continue_from_soma_flag):
                mask_path_from_soma[point] = 1
        # to tensor, and change to device
        mask_path = torch.from_numpy(mask_path).to(pred.device)
        mask_path_from_soma = torch.from_numpy(mask_path_from_soma).to(pred.device)

        ptls1 = (0.5 - pred) * mask_path
        ptls1 = torch.relu(ptls1)
        ptls2 = (1 - pred) * mask_path_from_soma

        pt_loss = pt_loss + torch.sum(ptls1) * torch.sum(ptls2)



    if(debug):
        temp_file_path = "/home/kfchen/temp_mip/" + str(time.time()) + ".png"
        mip_and_path_visualization(pred_clone, paths, temp_file_path, num_paths)


    return pt_loss
