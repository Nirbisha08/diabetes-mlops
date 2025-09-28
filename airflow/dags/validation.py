# validation.py

import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

def validate_raw_diabetes_dataset(csv_path):
    df = pd.read_csv(csv_path)

    context = gx.get_context()

    datasource_name = "default_pandas_datasource"
    suite_name = "raw_diabetes_suite"

    context.add_datasource(
        name=datasource_name,
        class_name="Datasource",
        execution_engine={"class_name": "PandasExecutionEngine"},
        data_connectors={
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"]
            }
        }
    )

    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="raw_diabetes_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "batch_raw_1"}
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    for col in df.columns:
        validator.expect_column_to_exist(col)
        validator.expect_column_values_to_not_be_null(col, mostly=0.95)

    validator.expect_column_values_to_be_between("Glucose", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("BloodPressure", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("SkinThickness", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("Insulin", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("BMI", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_in_set("Outcome", value_set=[0, 1])

    validator.save_expectation_suite()n
    validation_result = validator.validate()

    print("Validation success:", validation_result.success)
    print("Validation statistics:", validation_result.statistics)
    print("\n📋 Detailed Expectation Results:\n")

    for idx, result in enumerate(validation_result.results, 1):
        expectation_type = result.expectation_config.expectation_type
        kwargs = result.expectation_config.kwargs
        success = result.success

        print(f"🔹 Expectation {idx}: {expectation_type}")
        print(f"   ➤ Kwargs: {kwargs}")
        print(f"   ✅ Success: {success}")

        if not success:
            unexpected = result.result.get("unexpected_list", None)
            if unexpected:
                print(f"   ⚠️ Unexpected values: {unexpected}")
            elif "unexpected_percent" in result.result:
                print(f"   ⚠️ Unexpected %: {result.result['unexpected_percent']:.2f}%")
            elif "element_count" in result.result:
                print(f"   ⚠️ Details: {result.result}")
        print("-" * 80)

if __name__ == "__main__":
    validate_raw_diabetes_dataset("./artifacts/Messy_Healthcare_Diabetes.csv")