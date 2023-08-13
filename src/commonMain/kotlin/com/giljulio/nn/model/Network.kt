package com.giljulio.nn.model

class Network(vararg val layers: Layer) {
    fun forward(inputs: List<Double>): List<Double> {
        var currentInputs = inputs
        for (layer in layers) {
            currentInputs = layer.forward(currentInputs)
        }
        return currentInputs
    }
}
