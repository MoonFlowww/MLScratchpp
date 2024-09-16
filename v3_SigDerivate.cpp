#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <functional>

int train[][2] = {
    {0, 0}, {1, 1}, {2, 4}, {3, 9}, {4, 16},
    {5, 25}, {6, 36}, {7, 49}, {8, 64}, {9, 81},
    {10, 100}
};

#define train_count (sizeof(train) / sizeof(train[0]))

class NN {
private:
    int nInput, nOutput;
    std::vector<int> initHlayer;

    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;

    Eigen::VectorXd CurrLintput;

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> Activ_F;
    std::vector<Eigen::VectorXd> PreAct;
    std::vector<Eigen::VectorXd> Acted;
    double lr = 0.1;

    int ActParm;
    double c;

public:
    NN(int n_input, const std::vector<int>& hidden_layers, int n_output, int Parm)
        : nInput(n_input), initHlayer(hidden_layers), nOutput(n_output), ActParm(Parm) {
        std::srand(std::time(nullptr));
        c = (static_cast<double>(std::rand()) / RAND_MAX) / 10.0;

        std::vector<int> Arch = { nInput };
        for (auto& x : hidden_layers) Arch.push_back(x);
        Arch.push_back(n_output);

        for (int i = 1; i < Arch.size(); ++i) {
            weights.push_back(Eigen::MatrixXd::Random(Arch[i], Arch[i - 1]));
            biases.push_back(Eigen::VectorXd::Random(Arch[i]));
        }

        ActivFunc(ActParm);
    }


    void PrintParms() {
        std::cout << "\033[0mWeights: \n";
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << "Layer " << i + 1 << " weights:\n" << weights[i] << "\n";
        }
        std::cout << "\nBiases: \n";
        for (size_t i = 0; i < biases.size(); ++i) {
            std::cout << "Layer " << i + 1 << " biases:\n" << biases[i] << "\n";
        }
    }

    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
        return (1 / (1 + (-x.array()).exp())).matrix();
    }

    Eigen::VectorXd ReLu(const Eigen::VectorXd& x) {
        return x.array().max(0).matrix();
    }

    Eigen::VectorXd Tanh(const Eigen::VectorXd& x) {
        return x.array().tanh().matrix();
    }

    Eigen::VectorXd Msigmoid(const Eigen::VectorXd& x) {
        return (1 / (1 + (c * Eigen::VectorXd::Ones(x.size())).array().pow(-x.array()))).matrix();
    }

    Eigen::VectorXd PrimeMsig(const Eigen::VectorXd& x) {
        Eigen::VectorXd exp_neg_x = (c * Eigen::VectorXd::Ones(x.size())).array().pow(-x.array());
        Eigen::VectorXd one_plus_exp_neg_x = (1 + exp_neg_x.array()).matrix();
        Eigen::VectorXd term = (one_plus_exp_neg_x.array().pow(2)).matrix();
        return ((exp_neg_x.array() * std::log(c)) / term.array()).matrix();
    }


    Eigen::VectorXd DPrimeMsig(const Eigen::VectorXd& x) {
        Eigen::VectorXd exp_neg_x = (c * Eigen::VectorXd::Ones(x.size())).array().pow(-x.array());
        Eigen::VectorXd one_plus_exp_neg_x = (1 + exp_neg_x.array()).matrix();
        Eigen::VectorXd term1 = (one_plus_exp_neg_x.array().pow(3)).matrix();
        return ((-exp_neg_x.array() * std::log(c) * std::log(c)) / term1.array()).matrix();
    }






    void ActivFunc(int parm) {
        switch (parm) {
        default:
            Activ_F = [this](const Eigen::VectorXd& x) { return sigmoid(x); };
            break;
        case 1:
            Activ_F = [this](const Eigen::VectorXd& x) { return ReLu(x); };
            break;
        case 2:
            Activ_F = [this](const Eigen::VectorXd& x) { return Tanh(x); };
            break;
        case 3:
            Activ_F = [this](const Eigen::VectorXd& x) { return Msigmoid(x); };
            break;
        case 4:
            Activ_F = [this](const Eigen::VectorXd& x) { return PrimeMsig(x); };
            break;
        case 5:
            Activ_F = [this](const Eigen::VectorXd& x) { return DPrimeMsig(x); };
            break;
        }
    }

    void forward(const Eigen::VectorXd& x) {
        CurrLintput = x;

        for (unsigned i = 0; i < weights.size(); ++i) {
            Eigen::VectorXd preAct = weights[i] * CurrLintput + biases[i];
            PreAct.push_back(preAct);
            CurrLintput = Activ_F(preAct);
            Acted.push_back(CurrLintput);
        }
    }



    Eigen::VectorXd rForward(const Eigen::VectorXd& x) { // forward + return result outside of the class
        forward(x);
        return CurrLintput;
    }

    void backpropagate(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
        forward(input);

        Eigen::VectorXd output_error = Acted.back() - target;
        Eigen::VectorXd di = output_error.array() * Activ_F(PreAct.back()).array();

        for (int i = weights.size() - 1; i >= 0; --i) {
            Eigen::MatrixXd weight_grad = di * Acted[i - 1].transpose();
            Eigen::VectorXd bias_grad = di;

            weights[i] -= lr * weight_grad;
            biases[i] -= lr * bias_grad;

        }

        // error from "*"
        Eigen::VectorXd grad_c = -output_error.array() * (Eigen::VectorXd::Ones(Acted.back().size()).array() 
            / (1 + (c * Eigen::VectorXd::Ones(Acted.back().size()).array().pow(-Acted.back().array()))).array()).matrix(); 
        c -= lr * grad_c.mean();
    }


    double cost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y_target) {
        double MSE = 0;
        for (int i = 0; i < X.cols(); ++i) {
            forward(X.col(i));
            Eigen::VectorXd y_pred = CurrLintput; //this is the output of forward (func in void) 
            MSE += (y_pred - Y_target.col(i)).squaredNorm();
        }
        return (MSE / X.cols());
    }

    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, bool verbose = false) {
        for (int e = 0; e < epochs; ++e) {
            for (int i = 0; i < X.cols(); ++i) {
                backpropagate(X.col(i), Y.col(i));
            }
            if (verbose && ((e + 1) % 500 == 0 || e == epochs - 1)) {
                std::cout << "Epoch " << e + 1 << ", Cost: " << cost(X, Y) << std::endl;
            }
        }
    }
};

int main() {
    int total_epochs = 100000;
    std::vector<int> hidden_layers = { 10, 10 };

    NN nn(1, hidden_layers, 1, 3);

    Eigen::MatrixXd X_train(1, train_count);
    Eigen::MatrixXd Y_train(1, train_count);
    for (int i = 0; i < train_count; ++i) {
        X_train(0, i) = train[i][0] / 10.0;
        Y_train(0, i) = train[i][1] / 100.0;
    }

    std::cout << "\033[36m-> Training with the whole data, " << total_epochs << " epochs.\033[0m" << std::endl;
    nn.train(X_train, Y_train, total_epochs, true);
    nn.PrintParms();

    std::cout << "\n\n\n\033[0mPredictions after training:" << std::endl;
    Eigen::VectorXd input(1);
    Eigen::VectorXd y;
    for (int i = 0; i < train_count; ++i) {
        input << train[i][0] / 10.0;
        y = nn.rForward(input);
        std::cout << "\033[33mx: " << train[i][0] << "\033[0m, \033[96my: " << y(0) * 100.0
            << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0)
            << "\033[0m, \033[91mError: " << (train[i][1] - y(0) * 100.0) << "\033[0m" << std::endl;
    }

    // Test with new inputs
    input << 13 / 10.0;
    y = nn.rForward(input);
    std::cout << "\033[33mx: " << 13 << "\033[0m, \033[96my: " << y(0) * 100.0
        << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0)
        << "\033[0m, \033[91mError: " << (169 - y(0) * 100.0) << "\033[0m" << std::endl;

    input << 100.0 / 10.0;
    y = nn.rForward(input);
    std::cout << "\033[33mx: " << 100 << "\033[0m, \033[96my: " << y(0) * 100.0
        << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0)
        << "\033[0m, \033[91mError: " << (10000.0 - y(0) * 100.0) << "\033[0m" << std::endl;

    return 0;
}
