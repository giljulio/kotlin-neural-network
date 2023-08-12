package com.giljulio.nn

class Neuron(private val weights: List<Double>, private val bias: Double) {
    fun forward(inputs: List<Double>): Double {
        if (weights.size != inputs.size) {
            throw IllegalArgumentException("Number of weights and inputs must be the same")
        }

        var output = 0.0
        for (i in weights.indices) {
            output += weights[i] * inputs[i]
        }

        return output + bias
    }
}
