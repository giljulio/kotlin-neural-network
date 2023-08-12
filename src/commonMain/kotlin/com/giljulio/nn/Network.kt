package com.giljulio.nn

class Network(private val layers: List<Layer>) {
    fun forward(inputs: List<Double>): List<Double> {
        var currentInputs = inputs
        for (layer in layers) {
            currentInputs = layer.forward(currentInputs)
        }
        return currentInputs
    }
}
