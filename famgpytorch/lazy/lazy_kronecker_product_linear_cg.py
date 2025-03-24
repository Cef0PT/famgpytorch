#!/usr/bin/env python3

from linear_operator.operators import KroneckerProductLinearOperator, KroneckerProductDiagLinearOperator
from linear_operator.operators.diag_linear_operator import ConstantDiagLinearOperator


class KroneckerProductLinearOperatorLinearCG(KroneckerProductLinearOperator):
    def __add__(self, other):
        if isinstance(other, (KroneckerProductDiagLinearOperator, ConstantDiagLinearOperator)):
            from linear_operator.operators.added_diag_linear_operator import (
                AddedDiagLinearOperator,
            )
            return AddedDiagLinearOperator(self, other)
        return super().__add__(other)