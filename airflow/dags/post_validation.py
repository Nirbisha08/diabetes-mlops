import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

def validate_cleaned_diabetes_dataset(csv_path):
    # Load cleaned CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Load GE context from the local great_expectations/ directory
    context = gx.get_context()

    # Set expectation suite name
    suite_name = "postprocessed_diabetes_suite"

    # Create or overwrite suite
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    # Build a runtime batch request from the loaded DataFrame
    batch_request = RuntimeBatchRequest(
        datasource_name="default_pandas_datasource",
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="cleaned_diabetes_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "batch_cleaned_1"}
    )

    # Get validator and define expectations
    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)

    for col in df.columns:
        validator.expect_column_to_exist(col)
        validator.expect_column_values_to_not_be_null(col, mostly=0.95)

    validator.expect_column_values_to_be_between("Glucose", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("BloodPressure", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("SkinThickness", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("Insulin", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_between("BMI", min_value=1, mostly=0.95)
    validator.expect_column_values_to_be_in_set("Outcome", value_set=[0, 1])

    # Save and run validation
    validator.save_expectation_suite()
    validation_result = validator.validate()

    # Print result summary
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
    validate_cleaned_diabetes_dataset("./artifacts/diabetes_data_cleaned.csv")
