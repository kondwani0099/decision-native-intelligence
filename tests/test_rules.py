"""Unit tests for deterministic rule engines."""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rule_engine import FinanceRuleEngine, SalesRuleEngine, InventoryRuleEngine, evaluate_decision


class TestFinanceRuleEngine(unittest.TestCase):
    def test_equipment_exact_boundary(self):
        # 10,000 ZMW exact should approve
        ctx_approve = {
            "department": "operations",
            "vendor": "Lusaka Tech Supply",
            "expense_type": "equipment",
            "budget_remaining": 42000,
            "amount": 10000.0,
        }
        label, _ = FinanceRuleEngine.evaluate(ctx_approve)
        self.assertEqual(label, "approve")

        # 10,000.01 or 10,001 ZMW should review
        ctx_review = dict(ctx_approve, amount=10001.0)
        label, _ = FinanceRuleEngine.evaluate(ctx_review)
        self.assertEqual(label, "review")

    def test_non_equipment_category(self):
        ctx = {
            "department": "operations",
            "vendor": "Lusaka Tech Supply",
            "expense_type": "software_license",
            "budget_remaining": 42000,
            "amount": 15000.0,
        }
        label, _ = FinanceRuleEngine.evaluate(ctx)
        self.assertEqual(label, "approve")

    def test_missing_required_field(self):
        ctx = {
            "vendor": "Lusaka Tech Supply",
            "expense_type": "equipment",
            "budget_remaining": 42000,
            "amount": 8000.0,
        }
        label, _ = FinanceRuleEngine.evaluate(ctx)
        self.assertEqual(label, "review")


class TestSalesRuleEngine(unittest.TestCase):
    def test_discount_boundary(self):
        ctx_approve = {
            "customer_id": "CUST-ZAM-104",
            "customer_tier": "standard_retail",
            "order_value": 20000.0,
            "discount_percent": 15.0,
            "credit_limit": 60000.0,
            "outstanding_balance": 15000.0,
        }
        label, _ = SalesRuleEngine.evaluate(ctx_approve)
        self.assertEqual(label, "approve")

        ctx_review = dict(ctx_approve, discount_percent=15.1)
        label, _ = SalesRuleEngine.evaluate(ctx_review)
        self.assertEqual(label, "review")

    def test_credit_exposure_boundary(self):
        ctx_approve = {
            "customer_id": "CUST-ZAM-202",
            "customer_tier": "standard_retail",
            "order_value": 25000.0,
            "discount_percent": 10.0,
            "credit_limit": 50000.0,
            "outstanding_balance": 25000.0,  # total 50,000 <= 50,000
        }
        label, _ = SalesRuleEngine.evaluate(ctx_approve)
        self.assertEqual(label, "approve")

        ctx_review = dict(ctx_approve, outstanding_balance=25050.0)  # total 50,050 > 50,000
        label, _ = SalesRuleEngine.evaluate(ctx_review)
        self.assertEqual(label, "review")

    def test_enterprise_tier_bypass(self):
        ctx_enterprise = {
            "customer_id": "CUST-ZAM-ENT-1",
            "customer_tier": "tier_1_enterprise",
            "order_value": 40000.0,
            "discount_percent": 20.0,  # > 15% but <= 25% for tier 1
            "credit_limit": 100000.0,
            "outstanding_balance": 10000.0,
        }
        label, _ = SalesRuleEngine.evaluate(ctx_enterprise)
        self.assertEqual(label, "approve")


class TestInventoryRuleEngine(unittest.TestCase):
    def test_write_off_value_boundary(self):
        ctx_approve = {
            "item_code": "SKU-COP-440",
            "movement_type": "write_off",
            "quantity": 25,
            "unit_cost": 200.0,  # 5,000 ZMW exact
            "source_warehouse": "Kitwe Central Depot",
            "safety_stock_threshold": 100,
            "current_stock": 500,
        }
        label, _ = InventoryRuleEngine.evaluate(ctx_approve)
        self.assertEqual(label, "approve")

        ctx_review = dict(ctx_approve, unit_cost=201.0)  # 5,025 ZMW > 5,000
        label, _ = InventoryRuleEngine.evaluate(ctx_review)
        self.assertEqual(label, "review")

    def test_transfer_safety_stock_boundary(self):
        ctx_approve = {
            "item_code": "SKU-NDO-102",
            "movement_type": "transfer",
            "quantity": 80,
            "unit_cost": 150.0,
            "source_warehouse": "Ndola Main Warehouse",
            "safety_stock_threshold": 20,
            "current_stock": 100,  # 100 - 80 = 20 >= 20
        }
        label, _ = InventoryRuleEngine.evaluate(ctx_approve)
        self.assertEqual(label, "approve")

        ctx_review = dict(ctx_approve, quantity=81)  # 100 - 81 = 19 < 20
        label, _ = InventoryRuleEngine.evaluate(ctx_review)
        self.assertEqual(label, "review")


if __name__ == "__main__":
    unittest.main()
