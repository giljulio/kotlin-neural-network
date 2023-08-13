package com.giljulio.nn

import com.giljulio.nn.evaluator.MeanSquaredErrorEvaluator
import com.giljulio.nn.model.Layer
import com.giljulio.nn.model.Network
import com.giljulio.nn.model.Neuron
import com.giljulio.nn.trainer.Trainer
import com.giljulio.nn.util.randomBias
import com.giljulio.nn.util.randomWeight


fun main() {
    // Create a network with 2 inputs, 1 hidden layer with 2 neurons and 1 output
    val network = createNetwork()
    val evaluator = MeanSquaredErrorEvaluator()
    val trainer = Trainer(evaluator, network, iterations = 100000, learningRate = 0.2)

    // Training data for the AND function
    val trainingInputs = listOf(
        listOf(0.0, 0.0),
        listOf(0.0, 1.0),
        listOf(1.0, 0.0),
        listOf(1.0, 1.0)
    )
    val trainingTargets = listOf(
        listOf(0.0),
        listOf(0.0),
        listOf(0.0),
        listOf(1.0)
    )

    // train network
    trainer.trainAll(trainingInputs, trainingTargets)

    // Test the network
    val realData = listOf(
        listOf(0.0, 0.0),
        listOf(0.0, 1.0),
        listOf(1.0, 0.0),
        listOf(1.0, 1.0),
        listOf(2.0, 2.0),
        listOf(3.0, 7.0),
    )
    realData.forEachIndexed { i, data ->
        val output = network.forward(data)
        println("Inputs: $data, Expected: ${realData[i]}, Got: $output")
    }
}


/**
 * Creates a network with 2 inputs, 1 hidden layer with 2 neurons and 1 output
 */
private fun createNetwork(): Network {
    val inputLayer = Layer(
        Neuron(randomWeight(), randomWeight(), bias = randomBias()),
        Neuron(randomWeight(), randomWeight(), bias = randomBias()),
    )
    val hiddenLayer = Layer(
        Neuron(randomWeight(), randomWeight(), bias = randomBias()),
        Neuron(randomWeight(), randomWeight(), bias = randomBias()),
    )
    val outputLayer = Layer(
        Neuron(randomWeight(), randomWeight(), bias = randomBias())
    )

    return Network(inputLayer, hiddenLayer, outputLayer)
}
