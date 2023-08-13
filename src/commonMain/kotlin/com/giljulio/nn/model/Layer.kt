package com.giljulio.nn.model

class Layer(vararg val neurons: Neuron) {
    fun forward(inputs: List<Double>): List<Double> = neurons.map { neuron -> neuron.forward(inputs) }
}