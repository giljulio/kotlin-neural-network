package com.giljulio.nn

class Layer(private val neurons: List<Neuron>) {
    fun forward(inputs: List<Double>): List<Double> = neurons.map { neuron -> neuron.forward(inputs) }
}