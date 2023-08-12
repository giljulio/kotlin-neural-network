package com.giljulio.nn.evaluator

import com.giljulio.nn.Network

interface NetworkEvaluator {
    fun evaluate(network: Network, inputs: List<Double>): Double
}
