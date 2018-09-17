def rb_svd(mat, feature=20, steps = 100, gama = 1, lamda = 0.001):###train 0.00817, test 0.0147

    # feature是潜在因子的数量，mat为评分矩阵
    slowRate = 0.9
    nowRmse = 0.0

    user_feature = np.random.normal(0, .005, (mat.shape[0], feature))
    item_feature = np.random.normal(0, .005, (mat.shape[1], feature))
    bu = np.random.normal(0, .005, (mat.shape[0], 1))
    ##print bu.shape
    bi = np.random.normal(0, .005, (mat.shape[1], 1))
    print mat.shape[0]
    for step in range(steps):
        rmse = 0.0
        n = 0
        for u in range(510):
            for i in range(mat.shape[1]):
                if mat[u,i]>0:
                    pui = bu[u]+bi[i]+float(np.dot(user_feature[u, :], item_feature[i, :].T))
                    eui = mat[u, i] - pui
                    rmse += pow(eui, 2)
                    n += 1
                        # Rsvd的更新迭代公式
                    user_feature[u] += gama * (eui * item_feature[i] - lamda * user_feature[u])
                    item_feature[i] += gama * (eui * user_feature[u] - lamda * item_feature[i])
                    bu[u] +=gama*(eui- lamda*bu[u])
                    bi[i] +=gama*(eui-lamda*bi[i])

                        # n次迭代平均误差程度
        nowRmse = sqrt(rmse * 1.0 / n)
        print 'step: %d      Rmse: %s' % ((step + 1), nowRmse)
        if (nowRmse > preRmse):
            pass
        else:
            break
        # 降低迭代的步长
        gama *= slowRate
        step += 1
    pred_m = np.dot(user_feature,item_feature.T)
    for u in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            pred_m[u,i] = pred_m[u,i]+bu[u]+bi[i]
    return pred_m