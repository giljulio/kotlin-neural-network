package com.giljulio.nn

import com.giljulio.nn.evaluator.MeanSquaredErrorEvaluator
import com.giljulio.nn.model.Layer
import com.giljulio.nn.model.Network
import com.giljulio.nn.model.Neuron
import com.giljulio.nn.trainer.Trainer
import com.giljulio.nn.util.randomBias
import com.giljulio.nn.util.randomWeight
import kotlin.random.Random


fun main() {
    val network = createNetwork()
    val trainer = Trainer(
        evaluator = MeanSquaredErrorEvaluator(),
        network = network,
        iterations = 100000,
        learningRate = 0.01,
    )

    // Training data for the AND function
    val trainingInputs = List(100) { listOf(Random.nextInt(2).toDouble(), Random.nextInt(2).toDouble()) }
    val trainingTargets = trainingInputs.map { (a, b) -> listOf((a.toInt() and b.toInt()).toDouble()) }

    trainer.trainAll(trainingInputs, trainingTargets)

    // Test the network
    for (index in trainingInputs.indices) {
        val output = network.forward(trainingInputs[index])
        println("Inputs: ${trainingInputs[index]}, Expected: ${trainingTargets[index]}, Got: $output")
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
