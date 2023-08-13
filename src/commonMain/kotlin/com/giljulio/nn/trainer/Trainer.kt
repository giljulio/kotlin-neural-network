package com.giljulio.nn.trainer

import com.giljulio.nn.evaluator.NetworkEvaluator
import com.giljulio.nn.model.Network
import com.giljulio.nn.util.sigmoidDerivative

class Trainer(
    private val evaluator: NetworkEvaluator,
    private val network: Network,
    private val iterations: Int,
    private val learningRate: Double,
) {

    fun trainAll(inputs: List<List<Double>>, targets: List<List<Double>>) {
        if (targets.size != inputs.size) {
            throw IllegalArgumentException("Number of targets and inputs must be the same")
        }

        for (i in inputs.indices) {
            train(inputs[i], targets[i])
        }
    }

    fun train(inputs: List<Double>, targets: List<Double>): Network {
        for (i in 1..iterations) {
            // Use the evaluator to get the current error
            val error = evaluator.evaluate(network, inputs, targets)
            println("Iteration $i, error: $error")

            val outputs = network.forward(inputs)
            val errors = outputs.zip(targets).map { (o, t) -> o - t }

            // Start with the output error and derivatives
            var nextLayerErrors = errors.zip(outputs).map { (error, output) -> error * sigmoidDerivative(output) }

            // Start backpropagation from the last layer
            for (index in network.layers.indices.reversed()) {
                val layer = network.layers[index]
                val prevLayerOutputs =
                    if (index == 0) inputs else network.layers[index - 1].neurons.map { it.forward(inputs) }

                val currentLayerErrors = mutableListOf<Double>()

                for ((neuronIndex, neuron) in layer.neurons.withIndex()) {
                    val derivative = sigmoidDerivative(neuron.forward(prevLayerOutputs))

                    var neuronError = 0.0
                    // Here we loop through each neuron in the next layer (if exists)
                    if (index + 1 < network.layers.size) {
                        for (nextNeuronIndex in network.layers[index + 1].neurons.indices) {
                            val nextNeuron = network.layers[index + 1].neurons[nextNeuronIndex]
                            neuronError += nextLayerErrors[nextNeuronIndex] * nextNeuron.weights[neuronIndex] * derivative
                        }
                    } else {
                        neuronError = nextLayerErrors[neuronIndex]
                    }

                    currentLayerErrors.add(neuronError)

                    for (weightIndex in neuron.weights.indices) {
                        val weightError = neuronError * prevLayerOutputs[weightIndex]
                        neuron.weights[weightIndex] -= learningRate * weightError
                    }

                    neuron.bias -= learningRate * neuronError
                }

                nextLayerErrors = currentLayerErrors
            }
        }
        return network
    }
}