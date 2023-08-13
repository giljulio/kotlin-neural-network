package com.giljulio.nn.evaluator

import com.giljulio.nn.model.Network
import kotlin.math.pow

class MeanSquaredErrorEvaluator : NetworkEvaluator {
    override fun evaluate(network: Network, inputs: List<Double>, targets: List<Double>): Double {
        return network.forward(inputs).zip(targets).sumOf { (o, t) -> (o - t).pow(2) }
    }
}