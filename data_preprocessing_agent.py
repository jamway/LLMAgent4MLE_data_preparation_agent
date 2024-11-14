import os
from autogen import ConversableAgent,GroupChat, GroupChatManager
from functions import (handle_missing_values, 
handle_outliers, 
reduce_noise, 
merge_data_sources,
resolve_entity_conflicts,
normalize_data,
encode_categorical_data,
create_new_features,
check_data_quality)



def get_data_loader_prompt():
    return """Check if there is a need to load data, suggest to use the correct function how to load the data."""

def get_data_cleaner_prompt():
    return """Check if the data needs to be cleaned and suggest the best function to do proper cleaning."""
def get_data_integerator_prompt():
    return """Check if there is a need to integrate data sources, suggest the right function to merge data"""

def get_data_transformer_prompt():
    return """Check if data reqiures transformation, suggest to use the right functions."""

def get_feature_engineer_prompt():
    return """Depending on the request, suggest the function to create new features."""

def get_data_avalidator_prompt():
    return """Validate the data if needed."""

def get_data_profiler_prompt():
    return """Create a comprehensive report with data profiling in the way others can understand the data without accessing the full dataset."""

def get_chat_manager_prompt():
    return """You will be provided with the location of the data.
    The goal is to preprocess the data so that can be used for downstream feature engineering and machine learning training."""


def main():
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    
    # Data Loading Agent
    agent_data_loader = ConversableAgent(
        "agent_data_loader",
        system_message=get_data_loader_prompt(),
        llm_config=llm_config
    )
    # Data Cleaning Agent
    agent_data_cleaner = ConversableAgent(
        "agent_data_cleaner",
        system_instruction=get_data_cleaner_prompt(),
        llm_config=llm_config
    )
    agent_data_cleaner.register_for_llm(name="handle_missing_value", description="Impute missing value based on different strategy.")(handle_missing_values)
    agent_data_cleaner.register_for_llm(name="handle_outliers", description="Remove or process outlier from dataframe.")(handle_outliers)
    agent_data_cleaner.register_for_llm(name="reduce_noise", description="Reduce the noise in dataframe")(reduce_noise)

    # Data Integration Agent
    agent_data_integrator = ConversableAgent(
        "agent_data_itegrator",
        system_instruction=get_data_integerator_prompt(),
        llm_config=llm_config
    )
    agent_data_integrator.register_for_llm(name="merge_data_sources", description="Merge two data sources together by finding possible keys to join")(merge_data_sources)
    agent_data_integrator.register_for_llm(name="resolve_entity_conflicts", description="Resolve data conflict")(resolve_entity_conflicts)
    

    # Data Transformation Agent
    agent_data_transformer = ConversableAgent(
        "agent_data_transformer",
        system_instruction=get_data_transformer_prompt(),
        llm_config=llm_config
    )
    agent_data_transformer.register_for_llm(name="normalize_data", description="Normalizing data so that it can be used for certain machine leraning process.")(normalize_data)
    agent_data_transformer.register_for_llm(name="encode_categorical_data", description="Encode categorical data so that it can be used by machine learning model that can only process numerical data")(encode_categorical_data)

    # Feature Engineering Agent (Optional)
    agent_feature_engineer = ConversableAgent(
        "agent_feature_engineer",
        system_instruction=get_feature_engineer_prompt(),
        llm_config=llm_config
    )
    agent_feature_engineer.register_for_llm(name="create_new_features", description="Create new features for downstream model training step.")(create_new_features)

    # Data Validation Agent
    agent_data_validator = ConversableAgent(
        "agent_data_validator",
        system_instructions=get_data_avalidator_prompt(),
        functions=[
            # Implement functions to check data quality and consistency
            check_data_quality,
        ]
    )
    agent_data_validator.register_for_llm(name="check_data_quality", description="Check if the data quality is good for downstream machine learning model training step.")(check_data_quality)

    group_chat = GroupChat(
        agents=[agent_data_loader,agent_data_cleaner,agent_data_integrator,agent_data_transformer,agent_feature_engineer,agent_data_validator],
        messages=[],
        max_round=10
    )
    chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config
    )
    # TODO: Register all functions for execution
    chat_manager.register_for_execution(name="handle_missing_value")(handle_missing_values)
    chat_manager.register_for_execution(name="handle_outliers")(handle_outliers)
    chat_manager.register_for_execution(name="reduce_noise")(reduce_noise)
    chat_manager.register_for_execution(name="merge_data_sources")(merge_data_sources)
    chat_manager.register_for_execution(name="resolve_entity_conflicts")(resolve_entity_conflicts)
    chat_manager.register_for_execution(name="normalize_data")(normalize_data)
    chat_manager.register_for_execution(name="encode_categorical_data")(encode_categorical_data)
    chat_manager.register_for_execution(name="create_new_features")(create_new_features)
    chat_manager.register_for_execution(name="check_data_quality")(check_data_quality)

    chat_result = chat_manager.initiate_chat(
    group_chat_manager,
    message="Here is the data",
    summary_method="reflection_with_llm")

    print(chat_result)

if __name__ == "__main__":
    main()