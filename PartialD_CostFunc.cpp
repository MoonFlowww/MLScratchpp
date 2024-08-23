void gcost(double w1, double w2, double b, double *dw1, double *dw2, double *db) {
    *dw1 = 0.0;// pointer to redefined the value in the update func
    *dw2 = 0.0;
    *db = 0.0;


    for (size_t i = 0; i < train_count; ++i) {
        double x1 = train[i][0];
        double x2 = train[i][1];
        double z = train[i][2];
        double y = predict(x1, x2, w1, w2, b); // with sigmoid
        double di = 2 * (y - z) * y * (1 - y);
        *dw1 += di * x1;
        *dw2 += di * x2;
        *db += di;
            

    }
    *dw1 /=train_count;
    *dw2 /= train_count;
    *db /= train_count;
    
}




void update(int epoch) {
    srand(time(0));
    double w1 = (rand() / (double)RAND_MAX) * 10.0 -5.0;
    double w2 = (rand() / (double)RAND_MAX) * 10.0 - 5.0;
    double b = (rand() / (double)RAND_MAX) * 5.0 - 5.0;

    double eps = 1e-1;
    double rate = 1e-1;

    double init_cost = cost(w1, w2, b);
    double prev_c = init_cost;

    double backup_w1 = w1;
    double backup_w2 = w2;
    double backup_b = b;

    for (size_t e = 0; e < epoch; ++e) {
        /*                                                     // finite difference
        double c = cost(w1, w2, b);
        double dw1 = (cost(w1 + eps, w2, b) - c) / eps; 
        double dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        double db = (cost(w1, w2, b + eps) - c) / eps;
        */
        double dw1;
        double dw2;
        double db;

        gcost(w1, w2, b, &dw1, &dw2, &db); // ref to send the address to avoid unecessary variable copies
        
        w1 -= rate * dw1; // gradient
        w2 -= rate * dw2;
        b -= rate * db;

        double current_cost = cost(w1, w2, b);


        oss << "\n\n\nEpoch : " << e
            << "\n  w1 : " << w1
            << "\n  w2 : " << w2
            << "\n  b : " << b
            << "\n  cost :" << current_cost;

        std::cout << oss.str() << std::endl;
        oss.str("");

        if (current_cost < prev_c) {
            std::cout << "-> Better parameters found!!!" << std::endl;
            backup_w1 = w1;
            backup_w2 = w2;
            backup_b = b;
            prev_c = current_cost;
        }
    }
    double impr = ((init_cost - prev_c) / prev_c);
    std::cout << "\n\n\nTotal improvement : " << impr*100  << " %   ||   MPE : " << (impr/epoch)*100 << " %\n";
    // After the loop, if the backup parameters are better, restore them
    if (cost(backup_w1, backup_w2, backup_b) < cost(w1, w2, b)) {
        std::cout << "->Previous model is better than the last version!!! Restoring..." << std::endl;
        w1 = backup_w1;
        w2 = backup_w2;
        b = backup_b;
    }
    std::cout << "Or gate model ready !!" << std::endl;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("\n%zu  |  %zu = %f", i, j, predict(i, j, w1, w2, b));
        }
    }
}
