#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>

using namespace std;

struct Neuron {
    double value;
    vector<double> weights;
    double bias;
};

class Layer {
public:
    vector<Neuron> neurons;

    Layer(int neuronCount, int prevLayerNeuronCount) {
        neurons.resize(neuronCount);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1, 1);

        for (Neuron &neuron : neurons) {
            neuron.weights.resize(prevLayerNeuronCount);
            generate(neuron.weights.begin(), neuron.weights.end(), [&]() { return dis(gen); });
            neuron.bias = dis(gen);
        }
    }

    vector<double> getOutputs() {
        vector<double> outputs;
        for (const Neuron &neuron : neurons) {
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
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.emplace_back(layerSizes[i], layerSizes[i - 1]);
        }
    }

    // ReLU activation function
    double relu(double x) {
        return x > 0 ? x : 0;
    }

    // Derivative of the ReLU function
    double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    // Softmax activation function
    vector<double> softmax(const vector<double> &x) {
        vector<double> result(x.size());
        double maxElement = *max_element(x.begin(), x.end());
        double expSum = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = exp(x[i] - maxElement);
            expSum += result[i];
        }

        for (double &element : result) {
            element /= expSum;
        }

        return result;
    }

    // Forward propagation
    void forwardPropagation(const vector<double> &inputValues) {
        // Set input layer neuron values
        for (size_t i = 0; i < layers[0].neurons.size(); ++i) {
            layers[0].neurons[i].value = inputValues[i];
        }

        // Process hidden layers and output layer
        for (size_t i = 1; i < layers.size(); ++i) {
            for (Neuron &neuron : layers[i].neurons) {
                neuron.value = 0;
                for (size_t j = 0; j < layers[i - 1].neurons.size(); ++j) {
                    neuron.value += layers[i - 1].neurons[j].value * neuron.weights[j];
                }
                if (i != layers.size() - 1) {  // If not output layer, apply ReLU
                    neuron.value = relu(neuron.value + neuron.bias);
                } else {  // If output layer, add bias without ReLU
                    neuron.value += neuron.bias;
                }
            }
            // Apply Softmax to output layer
            if (i == layers.size() - 1) {
                vector<double> outputs = layers.back().getOutputs();
                outputs = softmax(outputs);
                for (size_t j = 0; j < layers.back().neurons.size(); ++j) {
                    layers.back().neurons[j].value = outputs[j];
                }
            }
        }
    }

    void backPropagation(const vector<double> &targetValues) {
        // Calculate output layer deltas
        vector<double> outputDeltas(layers.back().neurons.size());
        for (size_t i = 0; i < layers.back().neurons.size(); ++i) {
            outputDeltas[i] = (layers.back().neurons[i].value - targetValues[i]);
        }

        // Calculate hidden layer deltas
        vector<vector<double>> hiddenDeltas(layers.size() - 1);
        for (int i = layers.size() - 2; i >= 0; --i) {
            hiddenDeltas[i].resize(layers[i].neurons.size());
            for (size_t j = 0; j < layers[i].neurons.size(); ++j) {
                double sum = 0;
                for (size_t k = 0; k < layers[i + 1].neurons.size(); ++k) {
                    sum += layers[i + 1].neurons[k].weights[j] * (i == layers.size() - 2 ? outputDeltas[k] : hiddenDeltas[i + 1][k]);
                }
                hiddenDeltas[i][j] = sum * reluDerivative(layers[i].neurons[j].value);
            }
        }

        // Update weights and biases
        for (size_t i = layers.size() - 1; i > 0; --i) {
            for (size_t j = 0; j < layers[i].neurons.size(); ++j) {
                Neuron &neuron = layers[i].neurons[j];
                double delta = (i == layers.size() - 1) ? outputDeltas[j] : hiddenDeltas[i][j];
                for (size_t k = 0; k < neuron.weights.size(); ++k) {
                    neuron.weights[k] -= learningRate * delta * layers[i - 1].neurons[k].value;
                }
                neuron.bias -= learningRate * delta;
            }
        }
    }


    // Train function
    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                forwardPropagation(inputs[i]);
                backPropagation(targets[i]);
            }
        }
    }

    // Prediction function
    int predict(const vector<double> &input) {
        forwardPropagation(input);
        return distance(layers.back().neurons.begin(),
                                  max_element(layers.back().neurons.begin(), layers.back().neurons.end(),
                                              [](const Neuron &a, const Neuron &b) { return a.value < b.value; }));
 
    }

    // Evaluate the accuracy of the neural network
    double evaluateAccuracy(const vector<vector<double>> &inputs, const vector<vector<double>> &outputs) {
        int correctPredictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            forwardPropagation(inputs[i]);
            int predictedClass = distance(layers.back().neurons.begin(),
                                               max_element(layers.back().neurons.begin(), layers.back().neurons.end(),
                                                               [](const Neuron &a, const Neuron &b) { return a.value < b.value; }));
            int actualClass = distance(outputs[i].begin(), max_element(outputs[i].begin(), outputs[i].end()));
            if (predictedClass == actualClass) {
                correctPredictions++;
            }
        }
        return static_cast<double>(correctPredictions) / inputs.size();
    }

};


// Load the Iris dataset and split into training and validation sets
void loadIrisDataset(const string &filename, vector<vector<double>> &trainInputs, vector<vector<double>> &trainOutputs, vector<vector<double>> &validationInputs, vector<vector<double>> &validationOutputs, double trainSplit, double validationSplit) {
    // Load the entire dataset
    vector<vector<double>> inputs;
    vector<vector<double>> outputs;

    // The code for loading the dataset from a CSV file
    ifstream file(filename);
    string line;
    int lineNumber = 0;
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
                cerr << "Invalid value found at line " << lineNumber << ", column " << (i + 1) << ": " << value << endl;
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
        
    // Normalize the dataset
    vector<double> inputMeans(4, 0);
    vector<double> inputStds(4, 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputMeans[j] += inputs[i][j];
        }
    }
    for (size_t i = 0; i < inputMeans.size(); ++i) {
        inputMeans[i] /= inputs.size();
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputStds[j] += pow(inputs[i][j] - inputMeans[j], 2);
        }
    }
    for (size_t i = 0; i < inputStds.size(); ++i) {
        inputStds[i] = sqrt(inputStds[i] / inputs.size());
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputs[i][j] = (inputs[i][j] - inputMeans[j]) / inputStds[j];
        }
    } 

    // Randomly shuffle the dataset
    random_device rd;
    mt19937 g(rd());
    vector<size_t> indices(inputs.size());
    iota(indices.begin(), indices.end(), 0);

    shuffle(indices.begin(), indices.end(), g);

    // Split the dataset into training and validation sets
    size_t trainSize = static_cast<size_t>(inputs.size() * trainSplit);
    size_t validationSize = static_cast<size_t>(inputs.size() * validationSplit);

    for (size_t i = 0; i < trainSize; ++i) {
        trainInputs.push_back(inputs[indices[i]]);
        trainOutputs.push_back(outputs[indices[i]]);
    }

    for (size_t i = trainSize; i < trainSize + validationSize; ++i) {
        validationInputs.push_back(inputs[indices[i]]);
        validationOutputs.push_back(outputs[indices[i]]);
    }
    cout << "Training set size: " << trainInputs.size() << endl;
    cout << "Validation set size: " << validationInputs.size() << endl;

}


int main() {
    // Load the Iris dataset
    vector<vector<double>> trainInputs, trainOutputs, validationInputs, validationOutputs;
    loadIrisDataset("iris_dataset.csv", trainInputs, trainOutputs, validationInputs, validationOutputs, 0.90, 0.1);
    

    // Create neural network
    NeuralNetwork nn({4, 8, 128, 64, 8, 3},0.01); // Example: 4 input neurons, 2 hidden layers with 128 neurons each, and 3 output neurons

    // Train neural network
    nn.train(trainInputs, trainOutputs, 1000); // Train for 100 epochs

    // Test neural network
    double accuracy = nn.evaluateAccuracy(validationInputs, validationOutputs);
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;


    for (size_t i = 0; i < validationInputs.size(); ++i) {
        cout << "expected output:" << distance(validationOutputs[i].begin(), max_element(validationOutputs[i].begin(), validationOutputs[i].end())) << "\t";
        cout << "predicted output:" << nn.predict(validationInputs[i]) << endl;
    }

    return 0;
}

