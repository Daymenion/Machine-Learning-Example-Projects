//
// Neural Network in C++ from Scratch
// Created by: Daymenion
// MIT License
//

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <numeric>

using namespace std;

struct Neuron {
    double value;
    double bias;
    vector<double> weights;
};


class Layer {  
public:
    vector<Neuron> neurons;
    Layer(int neuronCount, int prevLayerNeuronCount) {
        neurons.resize(neuronCount);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1, 1);

        for(Neuron &neuron: neurons) {
            neuron.weights.resize(prevLayerNeuronCount);
            generate(neuron.weights.begin(), neuron.weights.end(), [&](){ return dis(gen); });
            neuron.bias = dis(gen);
        }
    }


    vector<double> getOutputs() {
        vector<double> outputs;
        for(Neuron &neuron: neurons) {
            outputs.push_back(neuron.value);
        }
        return outputs;
    }
};

class NeuralNetwork {
public:
    vector<Layer> layers;
    double learningRate;

    NeuralNetwork(const vector<int> &layerSizes, double learningRate)
        : learningRate(learningRate) {
        for(size_t i=1; i<layerSizes.size(); ++i) {
            layers.emplace_back(layerSizes[i], layerSizes[i-1]);
        }
    }

    double relu(double x) {
        return max(0.0, x);
    }

    double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }


    vector<double> softmax(const vector<double> &x) {
        vector<double> result(x.size());
        double maxElement = *max_element(x.begin(), x.end());
        double expSum = 0;

        for(size_t i=0; i<x.size(); ++i) {
            result[i] = exp(x[i] - maxElement);
            expSum += result[i];
        }
        for(double &element: result) {
            element /= expSum;
        }
        return result;
    }


    void forwardPropagation (const vector<double> &inputValues) {
        for(size_t i=0; i<layers[0].neurons.size(); ++i) {
            layers[0].neurons[i].value = inputValues[i];
        }

        for(size_t i=1; i<layers.size(); ++i) {
            for(Neuron &neuron: layers[i].neurons) {
                neuron.value = 0;
                for(size_t j=0; j<neuron.weights.size(); ++j) {
                    neuron.value += neuron.weights[j] * layers[i-1].neurons[j].value;
                }
                if(i!= layers.size()-1) {
                    neuron.value = relu(neuron.value + neuron.bias);
                }
                else {
                    neuron.value = neuron.value + neuron.bias;
                }
            }
            if(i==layers.size()-1) {
                vector<double> output = layers.back().getOutputs();
                output = softmax(output);
                for(size_t j=0; j<layers.back().neurons.size(); ++j) {
                    layers.back().neurons[j].value = output[j];
                }
            }
        }
    }

    void backProgpagation(const vector<double> &targetValues) {
        vector<double> outputDeltas(layers.back().neurons.size());
        for(size_t i=0; i<layers.back().neurons.size(); ++i) {
            outputDeltas[i] = layers.back().neurons[i].value - targetValues[i];
        }

        vector<vector<double>> hiddenDeltas(layers.size()-1);
        for(int i =layers.size()- 2; i>=0; --i) {
            hiddenDeltas[i].resize(layers[i].neurons.size());
            for(size_t j=0 ;j<layers[i].neurons.size(); ++j) {
                double sum = 0;
                for(size_t k=0; k<layers[i+1].neurons.size(); ++k) {
                    sum += layers[i+1].neurons[k].weights[j] * (i== layers.size()-2 ? outputDeltas[k] : hiddenDeltas[i+1][k]);
                }

                hiddenDeltas[i][j] = reluDerivative(layers[i].neurons[j].value) * sum;
            }
        } 

        for(size_t i =layers.size()-1; i>0; --i) {
            for(size_t j =0; j<layers[i].neurons.size(); ++j) {
                Neuron &neuron = layers[i].neurons[j];
                double delta = (i==layers.size()-1 ? outputDeltas[j] : hiddenDeltas[i-1][j]);

                for(size_t k=0; k<neuron.weights.size(); ++k) {
                    neuron.weights[k] -= learningRate * delta * layers[i-1].neurons[k].value;
                }
                neuron.bias -= learningRate * delta;
            }
        }
    }

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, int epochs) {
        for(int i=0; i<epochs; ++i) {
            for(size_t j=0; j<inputs.size(); ++j) {
                forwardPropagation(inputs[j]);
                backProgpagation(targets[j]);
            }
        }
    }

    int predict(const vector<double> &input) {
        forwardPropagation(input);

        return distance(layers.back().neurons.begin(), 
            max_element(layers.back().neurons.begin(), layers.back().neurons.end(), 
            [](const Neuron &a, const Neuron &b) { return a.value < b.value;}));
    }

    double evaluateAccuracy(const vector<vector<double>> &inputs, const vector<vector<double>> &targets) {
        int correctPredictions = 0;
        for(size_t i=0; i<inputs.size(); ++i) {
            forwardPropagation(inputs[i]);

            int predictedClass = distance(layers.back().neurons.begin(), 
                max_element(layers.back().neurons.begin(), layers.back().neurons.end(), 
                [](const Neuron &a, const Neuron &b) { return a.value < b.value;}));
            
            int actualClass = distance(targets.begin(), max_element(targets.begin(), targets.end()));

            if(predictedClass == actualClass) {
                ++correctPredictions;
            }
        }

        return static_cast<double>(correctPredictions) / inputs.size();
    }

};

void loadIrsihDataset(const string &filename, vector<vector<double>> &trainInputs,
    vector<vector<double>> &trainOutputs, vector<vector<double>> &validationsInputs,
    vector<vector<double>> &validationOutputs, double trainSplit, double validationSplit) {


        vector<vector<double>> inputs;
        vector<vector<double>> outputs;

        ifstream file(filename);
        string line;
        int lineNumber=0;
    while (getline(file, line)) {
        lineNumber++;
        istringstream lineStream(line);
        vector<double> input(4);
        vector<double> output(3, 0);

        for (size_t i = 0; i < 4; ++i) {
            string value;
            getline(lineStream, value, ',');

            if (value.empty()) {
                cerr << "Empty value found at line " << lineNumber << ", column " << (i + 1) << endl;
                continue;
            }

            try {
                input[i] = stod(value);
            } catch (const invalid_argument &e) {
                cerr << "Invalid value found at line " << 
                lineNumber << ", column " << (i + 1) << ": " << value << endl;
                continue;
            }
        }

        string label;
        getline(lineStream, label);
        if (label == "Iris-setosa") {
            output[0] = 1;
        } else if (label == "Iris-versicolor") {
            output[1] = 1;
        } else if (label == "Iris-virginica") {
            output[2] = 1;
        } else {
            cerr << "Invalid label found at line " << lineNumber << ": " << label << endl;
            continue;
        }

        inputs.push_back(input);
        outputs.push_back(output);
    }


        vector<double> inputMeans(4,0);
        vector<double> inputStds(4,0);

        for(size_t i=0; i< inputs.size(); ++i) {
            for(size_t j =0; j<inputs.size(); ++j)
                inputMeans[j] += inputs[i][j];
        }
        
            printf("burada1");
        for(size_t i =0;i<inputMeans.size();++i)
            inputMeans[i] /= inputs.size();

        for(size_t i=0; i< inputs.size(); ++i) {
            for(size_t j =0; j<inputs.size(); ++j)
                inputStds[j] += pow(inputs[i][j] - inputMeans[j], 2);
        }

        for(size_t i =0;i<inputStds.size();++i)
            inputStds[i] = sqrt(inputStds[i] / inputs.size());
        
            printf("burada2");
        for(size_t i=0; i< inputs.size(); ++i) {
            for(size_t j =0; j<inputs.size(); ++j)
                inputs[i][j] = (inputs[i][j] - inputMeans[j]) / inputStds[j];
        }
        
            printf("burada3");
        random_device rd;
        mt19937 g(rd());
        vector<size_t> indices(inputs.size());
        iota(indices.begin(), indices.end() , 0);

        shuffle(indices.begin(), indices.end(),g);

        size_t trainsize = static_cast<size_t>(inputs.size() * trainSplit);
        size_t validationSize = static_cast<size_t>(inputs.size() * validationSplit);

        for(size_t i=0; i< trainsize; ++i) {
            trainInputs.push_back(inputs[indices[i]]);
            trainOutputs.push_back(outputs[indices[i]]);
        }
        for(size_t i=trainsize; i< trainsize + validationSize; ++i) {
            validationsInputs.push_back(inputs[indices[i]]);
            validationOutputs.push_back(outputs[indices[i]]);
        }

        cout << "Train size: " << trainInputs.size() << endl;
        cout << "Validation size: " << validationsInputs.size() << endl;
}


int main() {
    vector<vector<double>> trainInputs, trainOutputs, validationInputs, validationOutputs;
    loadIrsihDataset("iris_dataset.csv", trainInputs, trainOutputs, validationInputs, validationOutputs, 0.9, 0.1);

    NeuralNetwork nn({4, 5, 3}, 0.01);
    nn.train(trainInputs, trainOutputs, 100);
    double accuracy = nn.evaluateAccuracy(validationInputs, validationOutputs);
    cout << "Accuracy: " << accuracy*100 << "%" << endl;

    for(size_t i =0; i<validationInputs.size(); ++i) {
        cout << "ecpected output:" << distance(validationOutputs[i].begin(), max_element(validationOutputs[i].begin(), validationOutputs[i].end())) << "\t";
        cout << "predicted output:" << nn.predict(validationInputs[i]) << endl;
    }

    return 0;
    
}