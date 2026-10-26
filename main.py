import argparse
import asyncio
import contextvars
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
from aggregation import aggregate_binary

EXTREMIZE = float(os.getenv("EXTREMIZE") or 1.0)
SPEND_STOP = float(os.getenv("SPEND_STOP") or 90)  # USD/month: halt everything
SPEND_NO_REFORECAST = float(os.getenv("SPEND_NO_REFORECAST") or 85)  # USD/month: stop re-forecasts
logger = logging.getLogger(__name__)


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


# USD per million tokens (input, output) for the raw web-search call, which litellm does not track.
SEARCH_PRICES = {"haiku-5-5": (0.10, 0.50), "sonnet-5-5": (2.0, 10.0), "opus-5-5": (4.0, 20.0)}
_SLOT = contextvars.ContextVar("prediction_slot", default=None)


def _tier_env(name: str, tier: str) -> str | None:
    """Tier 'main' (FutureEval Fall 33121, the high-value questions) reads NAME_MAIN first."""
    if tier == "main":
        v = os.getenv(name + "_MAIN")
        if v:
            return v
    return os.getenv(name)


class MoneyBot(FallTemplateBot2026):
    """Improved bot: multi-model ensemble, log-odds aggregation + moderate extremization,
    outside-view binary prompt. Models come from env FORECAST_MODELS (comma list, LiteLLM
    names) so they can be changed without code. Two tiers: 'main' (Fall tournament, env *_MAIN
    overrides) and 'base' (MiniBench, Market Pulse)."""

    _rr = 0
    _search_uses = 0
    _search_usd = 0.0
    _tier = "base"

    def set_tier(self, tier: str) -> None:
        self._tier = tier
        self.predictions_per_research_report = int(_tier_env("PREDICTIONS_PER_REPORT", tier) or 6)
        print(f"TIERLOG tier={tier} runs={self._run_plan()} search={self._search_cfg()}")

    def _ensemble(self) -> list[GeneralLlm]:
        cache = self.__dict__.setdefault("_ens_by_tier", {})
        if self._tier not in cache:
            names = [m.strip() for m in (_tier_env("FORECAST_MODELS", self._tier) or "").split(",") if m.strip()]
            cache[self._tier] = [GeneralLlm(model=n, temperature=0.3, timeout=300, allowed_tries=2) for n in names]
        return cache[self._tier]

    def _run_plan(self) -> str:
        """Which model runs each prediction slot of a question, e.g. 4x claude-opus-5-5 + 2x claude-sonnet-5-5."""
        ens = self._ensemble()
        if not ens:
            return "default model"
        from collections import Counter
        c = Counter(ens[i % len(ens)].model.split("/")[-1] for i in range(self.predictions_per_research_report))
        return " + ".join(f"{n}x {m}" for m, n in c.items())

    def _search_cfg(self) -> tuple[str, int]:
        model = (_tier_env("SEARCH_MODEL", self._tier) or "claude-haiku-5-5").removeprefix("anthropic/")
        return model, int(_tier_env("SEARCH_MAX_USES", self._tier) or 2)

    async def _make_prediction(self, question, research):
        # Deterministic model per prediction slot (slot i -> ensemble[i % len]); each prediction runs in
        # its own asyncio task, so the context variable is local to it.
        counts = self.__dict__.setdefault("_slot_counts", {})
        key = (question.page_url, getattr(question, "id_of_question", None), getattr(question, "conditional_type", None))
        slot = counts.get(key, 0)
        counts[key] = slot + 1
        _SLOT.set(slot)
        return await super()._make_prediction(question, research)

    def get_llm(self, purpose="default", guarantee_type=None):
        if purpose == "default" and guarantee_type == "llm" and self._ensemble():
            ens = self._ensemble()
            slot = _SLOT.get()
            if slot is None:
                MoneyBot._rr += 1
                slot = MoneyBot._rr
            return ens[slot % len(ens)]
        return super().get_llm(purpose, guarantee_type)

    async def run_research(self, question):
        """Research with fallbacks: configured researcher -> Anthropic web search -> no research.
        A failing search provider (e.g. AskNews credits exhausted) must never stop forecasting."""
        mode = os.getenv("RESEARCH_MODEL") or "anthropic-search"
        if mode == "anthropic-search":
            try:
                return await self._anthropic_search(question)
            except Exception as e:
                logger.warning(f"anthropic-search failed ({e!r}); continuing without research")
                return ""
        try:
            return await super().run_research(question)
        except Exception as e:
            logger.warning(f"researcher {mode} failed ({e!r}); falling back to anthropic-search")
        try:
            return await self._anthropic_search(question)
        except Exception as e:
            logger.warning(f"anthropic-search failed ({e!r}); continuing without research")
            return ""

    async def _anthropic_search(self, question) -> str:
        import requests
        model, max_uses = self._search_cfg()
        prompt = clean_indents(
            f"""
            You assist a superforecaster. Search the web for the most relevant and RECENT facts for this question
            (current status and latest values, base rates / historical frequencies, market prices, polls, expert forecasts,
            scheduled events before the resolution date). Today is {datetime.now().strftime("%Y-%m-%d")}.
            Write a thorough bulleted rundown (about 300-600 words) with dates and sources, ending with the 3 facts that
            matter most for the resolution. Do NOT give a probability.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            {question.fine_print}
            """
        )
        # Newer models get the dynamic-filtering search tool; Haiku keeps the basic one.
        tool_type = "web_search_20260209" if ("sonnet" in model or "opus" in model) else "web_search_20250305"
        messages = [{"role": "user", "content": prompt}]
        headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"}

        def _call(msgs):
            body = {
                "model": model,
                "max_tokens": 8000,  # thinking shares this budget; 1500 left some answers almost empty
                "messages": msgs,
                "tools": [{"type": tool_type, "name": "web_search", "max_uses": max_uses}],
            }
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=300)
            r.raise_for_status()
            return r.json()

        uses, usd, data = 0, 0.0, {}
        pin, pout = next((v for k, v in SEARCH_PRICES.items() if k in model), (4.0, 20.0))
        async with self._concurrency_limiter:
            for _ in range(3):  # server tools may stop with pause_turn: continue the same turn
                data = await asyncio.to_thread(_call, messages)
                u = data.get("usage", {}) or {}
                uses += int((u.get("server_tool_use") or {}).get("web_search_requests", 0) or 0)
                usd += ((u.get("input_tokens") or 0) + 1.25 * (u.get("cache_creation_input_tokens") or 0)
                        + 0.1 * (u.get("cache_read_input_tokens") or 0)) * pin / 1e6 + (u.get("output_tokens") or 0) * pout / 1e6
                if data.get("stop_reason") != "pause_turn":
                    break
                messages = messages + [{"role": "assistant", "content": data.get("content", [])}]
        blocks = data.get("content", []) or []
        last_tool = max([i for i, b in enumerate(blocks) if b.get("type", "").endswith("tool_result")] or [-1])
        text = "\n".join(b.get("text", "") for b in blocks[last_tool + 1:] if b.get("type") == "text").strip()
        if len(text) < 200:
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        MoneyBot._search_uses += uses
        MoneyBot._search_usd += usd
        logger.info(f"SEARCHLOG anthropic-search model={model} uses={uses} chars={len(text)} tok_usd={usd:.4f} stop={data.get('stop_reason')} url={question.page_url}")
        return text

    async def _run_forecast_on_binary(self, question, research):
        prompt = clean_indents(
            f"""
            You are a calibrated superforecaster. Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write, briefly:
            (a) Time until resolution.
            (b) Outside view: base rate for this kind of event in the reference class.
            (c) Status quo outcome and how likely it persists.
            (d) Best scenario for Yes, best scenario for No.
            (e) Market/expert signals in the research, if any.
            (f) Inside-view adjustment from the base rate, with the key evidence.
            Be well calibrated: avoid <3% or >97% unless the outcome is nearly certain by the rules.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    def _create_unified_explanation(self, question, research_prediction_collections, aggregated_prediction, final_cost, time_spent_in_minutes):
        text = super()._create_unified_explanation(
            question, research_prediction_collections, aggregated_prediction, final_cost, time_spent_in_minutes
        )
        smodel, suses = self._search_cfg()
        note = (
            "\n\n---\n"
            f"Method note: research by {smodel} with up to {suses} web searches; then {self.predictions_per_research_report} "
            f"independent forecasting runs ({self._run_plan()}). "
            f"Binary answers are combined as a mean of log-odds (dropping the min and max when there are 5+ runs) "
            f"with extremization factor {EXTREMIZE}; numeric/date answers use the median CDF and multiple choice the "
            "standard forecasting-tools aggregation. The published number is the aggregate, not any single run. "
            "No human reviewed or edited this forecast."
        )
        return text + note

    async def _aggregate_predictions(self, predictions, question):
        if isinstance(question, BinaryQuestion):
            return aggregate_binary([float(p) for p in predictions], EXTREMIZE)
        return await super()._aggregate_predictions(predictions, question)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions", "diag", "report", "mp", "review"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions", "diag", "report", "mp", "review"] = args.mode
    if run_mode == "review":  # read-only: scores of resolved questions, no LLM, no publishing
        import review_mode
        review_mode.run()
        raise SystemExit(0)
    if run_mode == "report":
        import requests as _rq
        _h = {"Authorization": "Token " + os.environ["METACULUS_TOKEN"]}
        for _pid in ["33121", "33132"]:
            try:
                _r = _rq.get(f"https://www.metaculus.com/api/leaderboards/project/{_pid}/", headers=_h, timeout=60)
                _data = _r.json() if _r.status_code == 200 else []
                for _lb in _data:
                    _es = _lb.get("entries") or []
                    _me = [(i + 1, e) for i, e in enumerate(_es) if "freaum" in str(e).lower()]
                    if _es:
                        print("REPORT", _pid, "top1", str(_es[0])[:300])
                    print("REPORT", _pid, _lb.get("project_name"), "score_type", _lb.get("score_type"), "finalized", _lb.get("finalized"),
                          "end", _lb.get("end_time"), "prize_pool", _lb.get("prize_pool"), "entries", len(_es),
                          "me", str(_me)[:300])
                if _r.status_code != 200:
                    print("REPORT", _pid, "HTTP", _r.status_code)
            except Exception as _e:
                print("REPORT ERR", _pid, repr(_e)[:150])
        try:
            _qs = MetaculusClient().get_all_open_questions_from_tournament("minibench")
            if _qs:
                _aj = getattr(_qs[0], "api_json", {}) or {}
                _pj = _aj.get("projects") or (_aj.get("post", {}) or {}).get("projects")
                print("REPORT minibench_projects", str(_pj)[:500])
            else:
                print("REPORT minibench_projects no open questions")
        except Exception as _e:
            print("REPORT ERR minibench_probe", repr(_e)[:150])
        raise SystemExit(0)
    if run_mode == "diag":
        import asyncio as _a
        _c = MetaculusClient()
        for _t in ["minibench", 33132, 33121, "fall-futureeval-2026", "bot-testing-area", "market-pulse-26q4"]:
            try:
                _qs = _c.get_all_open_questions_from_tournament(_t)
                print("DIAG", _t, len(_qs), [(q.page_url, type(q).__name__, str(getattr(q, "close_time", ""))[:16], getattr(q, "group_question_option", None)) for q in _qs[:6]])
            except Exception as _e:
                print("DIAG", _t, "ERR", repr(_e)[:200])
        raise SystemExit(0)

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. The `llms=` block below is commented out to use
    # whichever default models forecasting-tools picks based on your env vars;
    # uncomment and edit to pin specific models.
    template_bot = MoneyBot(
        research_reports_per_question=1,
        predictions_per_research_report=int(os.getenv("PREDICTIONS_PER_REPORT") or 6),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        enable_summarize_research=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(model=os.getenv("DEFAULT_MODEL") or "anthropic/claude-haiku-5-5", temperature=0.3, timeout=240, allowed_tries=2),
            "summarizer": os.getenv("AUX_MODEL") or "anthropic/claude-haiku-5-5",
            "researcher": os.getenv("RESEARCH_MODEL") or "anthropic/claude-haiku-5-5",
            "parser": os.getenv("AUX_MODEL") or "anthropic/claude-haiku-5-5",
        },
        # llms (template example, kept for reference):
        # llms={
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o",
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/news-summaries",
        #     "parser": "openai/gpt-4o-mini",
        # },
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode in ("tournament", "mp"):
        _spend = float(os.getenv("SPEND_TOTAL") or 0)
        if _spend >= SPEND_STOP:
            print(f"SPENDLOG halted: month total {_spend:.2f} >= {SPEND_STOP}")
            raise SystemExit(0)
        if _spend >= SPEND_NO_REFORECAST:
            print(f"SPENDLOG re-forecasts off: month total {_spend:.2f} >= {SPEND_NO_REFORECAST}")
            os.environ["REFORECAST_HOURS"] = "0"

        async def _both():
            a, b = [], []
            if run_mode == "tournament":
                template_bot.set_tier("main")  # Fall tournament: highest prize per question
                a = await template_bot.forecast_on_tournament(
                    client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
                template_bot.set_tier("base")
                b = await template_bot.forecast_on_tournament(
                    client.CURRENT_MINIBENCH_ID, return_exceptions=True
                )
            template_bot.set_tier("base")
            c = []
            mp_id = os.getenv("MARKET_PULSE_ID")  # e.g. market-pulse-26q4; empty = off
            if mp_id:
                # Market Pulse needs forecasts kept up to date: re-forecast every REFORECAST_HOURS
                # (empty/0 = forecast each question only once).
                hours = float(os.getenv("REFORECAST_HOURS") or 0)
                now = datetime.now(timezone.utc)
                due = []
                for q in client.get_all_open_questions_from_tournament(mp_id):
                    last = None
                    pf = q.previous_forecasts
                    if pf:
                        last = pf[-1].timestamp
                    else:
                        try:  # fallback: some question types do not parse the history
                            mf = q.api_json["question"]["my_forecasts"]
                            st = (mf.get("latest") or {}).get("start_time") or (mf.get("history") or [{}])[-1].get("start_time")
                            if st:
                                last = datetime.fromtimestamp(st, tz=timezone.utc)
                        except Exception:
                            last = None
                    if last is None or (hours and (now - last).total_seconds() >= hours * 3600):
                        due.append(q)
                print(f"MPLOG id={mp_id} due={len(due)} hours={hours} due_ids={[q.id_of_question for q in due][:30]}")
                if due:
                    template_bot.skip_previously_forecasted_questions = False
                    try:
                        c = await template_bot.forecast_questions(due, return_exceptions=True)
                    finally:
                        template_bot.skip_previously_forecasted_questions = True
            return a + b + c

        try:
            _open = {
                t: len(client.get_all_open_questions_from_tournament(t))
                for t in (client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID)
            }
        except Exception as _e:
            _open = f"err {_e!r}"[:100]
        forecast_reports = asyncio.run(_both())
        _ok = [r for r in forecast_reports if not isinstance(r, Exception)]
        print(f"RUNLOG utc={datetime.now(timezone.utc).isoformat(timespec='minutes')} open={_open} reports_ok={len(_ok)} total={len(forecast_reports)}")  # single event loop (semaphore is class-level)
        # litellm tracks the forecasting calls; the raw web-search call is added here (search fees + its tokens)
        _cost = sum((getattr(r, "price_estimate", None) or 0) for r in _ok) + MoneyBot._search_uses * 0.01 + MoneyBot._search_usd
        print(f"COSTLOG usd={_cost:.4f} reports={len(_ok)} searches={MoneyBot._search_uses} search_tok_usd={MoneyBot._search_usd:.4f}")
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        template_bot.set_tier(os.getenv("TEST_TIER") or "base")
        _tq = client.get_all_open_questions_from_tournament("bot-testing-area")
        _lim = int(os.getenv("TEST_LIMIT") or 0)
        if _lim:  # one question of each type first, to keep test cost low
            _by = {}
            for _q in _tq:
                _by.setdefault(type(_q).__name__, []).append(_q)
            _order = [q for i in range(max(len(v) for v in _by.values())) for v in _by.values() if i < len(v)]
            _tq = _order[:_lim]
        print("TESTLOG questions", [(q.page_url, type(q).__name__) for q in _tq])
        forecast_reports = asyncio.run(template_bot.forecast_questions(_tq, return_exceptions=True))
        _ok = [r for r in forecast_reports if not isinstance(r, Exception)]
        for _r in _ok:
            print(f"TESTLOG cost {getattr(_r, 'price_estimate', 0) or 0:.4f} {_r.question.page_url} {type(_r.question).__name__}")
        _cost = sum((getattr(r, "price_estimate", None) or 0) for r in _ok) + MoneyBot._search_uses * 0.01 + MoneyBot._search_usd
        print(f"COSTLOG usd={_cost:.4f} reports={len(_ok)} total={len(forecast_reports)} searches={MoneyBot._search_uses} search_tok_usd={MoneyBot._search_usd:.4f}")

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
