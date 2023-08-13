package com.giljulio.nn.util

import kotlin.math.exp

fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

fun sigmoidDerivative(x: Double): Double = sigmoid(x) * (1 - sigmoid(x))