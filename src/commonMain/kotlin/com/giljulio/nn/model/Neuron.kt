package com.giljulio.nn.model

class Neuron(val weights: MutableList<Double>, var bias: Double) {

    constructor(vararg weights: Double, bias: Double) : this(weights.toMutableList(), bias)

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
