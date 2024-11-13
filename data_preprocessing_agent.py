from swarm import Swarm, Agent
from functions import (handle_missing_values, 
handle_outliers, 
reduce_noise, 
merge_data_sources,
resolve_entity_conflicts,
normalize_data,
encode_categorical_data,
create_new_features,
check_data_quality)


client = Swarm()

def main():

    # Data Loading Agent
    agent_data_loader = Agent(
        name="Agent Data Loader"
        instruction="I can "
    )

    # Data Cleaning Agent
    agent_data_cleaner = Agent(
        name="Agent Data Cleaner",
        instructions="I clean and prepare your data. Tell me if you have missing values or outliers.",
        functions=[
            # Implement functions to handle missing values, outliers, and noise reduction
            handle_missing_values,
            handle_outliers,
            reduce_noise,
        ]
    )

    # Data Integration Agent
    agent_data_integrator = Agent(
        name="Agent Data Integrator",
        instructions="I combine data from different sources. Provide me with the data sources.",
        functions=[
            # Implement functions to merge data sources and resolve entity conflicts
            merge_data_sources,
            resolve_entity_conflicts,
        ]
    )

    # Data Transformation Agent
    agent_data_transformer = Agent(
        name="Agent Data Transformer",
        instructions="I transform and prepare your data for modeling. Do you need normalization or encoding?",
        functions=[
            # Implement functions to normalize and encode data
            normalize_data,
            encode_categorical_data,
        ]
    )

    # Feature Engineering Agent (Optional)
    agent_feature_engineer = Agent(
        name="Agent Feature Engineer",
        instructions="I create new features to improve model performance. Tell me about your data and the target variable.",
        functions=[
            # Implement functions to create new features
            create_new_features,
        ]
    )

    # Data Validation Agent
    agent_data_validator = Agent(
        name="Agent Data Validator",
        instructions="I ensure your data is clean and ready for use. Provide me with your data.",
        functions=[
            # Implement functions to check data quality and consistency
            check_data_quality,
        ]
    )
    def transfer_to_agent_data_loader():
        return agent_data_loader
    def transfer_to_agent_data_profiler():
        return agent_data_profiler
    def transfer_to_agent_data_integrator():
        return agent_data_integrator
    def transfer_to_agent_data_cleaner():
        return agent_data_cleaner
    def transfer_to_agent_data_integrator():
        return agent_data_integrator
    def transfer_to_agent_data_transformer():
        return agent_data_transformer
    def transfer_to_agent_feature_engineer():
        return agent_feature_engineer  # Optional
    def transfer_to_agent_data_validator():
        return agent_data_validator
        # Main Agent
    agent_a = Agent(
        name="Agent A",
        instructions="""To perform a successful machine learning project, 
        you need to preprocess the data for feature engineering or for model to consume. 
        Model inforation might be available
        Focus on preprocessing the data to make it available for the most machine learning model to consume.
        """,
        functions=[
            transfer_to_agent_data_loader,
            transfer_to_agent_data_profiler,
            transfer_to_agent_data_cleaner,
            transfer_to_agent_data_integrator,
            transfer_to_agent_data_transformer,
            transfer_to_agent_feature_engineer,  # Optional
            transfer_to_agent_data_validator,
        ]
    )

    response = client.run(
        agent=agent_a,
        messages=[{"role": "user", "content": "I have the dataframe."}],
    )
    print(response.messages[-1]["content"])

if __name__ == "__main__":
    main()