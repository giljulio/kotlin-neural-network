package com.giljulio.nn.evaluator

import com.giljulio.nn.model.Network

interface NetworkEvaluator {
    fun evaluate(network: Network, inputs: List<Double>, targets: List<Double>): Double
}
