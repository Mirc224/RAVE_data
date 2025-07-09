from models.rave_dataset import RaveSample
from collections import defaultdict
from experiments.running_experiments import SampleProcessingResult
from pydantic_core import from_json

class SampleResponse:
    def __init__(self, response: dict):
        self._prompt_tokens_used:int = response[SampleProcessingResult.USED_PROMPT_TOKENS]
        self._completion_tokens_used: int = response[SampleProcessingResult.USED_COMPLETION_TOKENS]
        self._total_tokens_used: int = response[SampleProcessingResult.TOTAL_USED_TOKENS]
        self._has_valid_json: bool = True
        self._parsed_json: dict = {}
        self._raw_response: str = response[SampleProcessingResult.MODEL_RESPONSE]

        self.__try_parse_response(self._raw_response)
    
    def __try_parse_response(self, raw_response: str):
        try:
            # parsed_value = json.loads(raw_response)
            parsed_value = from_json(raw_response, allow_partial=False)
            self._parsed_json = parsed_value
        except:
            self._has_valid_json = False

    @property
    def prompt_tokens_used(self) -> int:
        return self._prompt_tokens_used
    
    @property
    def completion_tokens_used(self) -> int:
        return self._completion_tokens_used
    
    @property
    def total_token_used(self) -> int:
        return self._total_tokens_used
    
    @property
    def parsed_json(self) -> dict:
        return self._parsed_json
    
    @property
    def raw_response(self) -> str:
        return self._raw_response
    
    @property
    def has_valid_json(self) -> bool:
        return self._has_valid_json

class SampleResult:
    def __init__(self, sample: dict):
        self._sample_id: int = sample[SampleProcessingResult.ID]
        self._start_time: float = sample[SampleProcessingResult.START_TIME]
        self._end_time: float = sample[SampleProcessingResult.END_TIME]
        self._responses: list[SampleResponse] = self.__parse_sample_responses(sample[SampleProcessingResult.RESULT_RESPONSES])

    def __parse_sample_responses(self, raw_responses: list[dict]) -> list[SampleResponse]:
        result_list: list[SampleResponse] = []
        for raw_response in raw_responses:
            processed_response = SampleResponse(raw_response)
            result_list.append(processed_response)
        return result_list

    @property
    def sample_id(self) -> int:
        return self._sample_id
    
    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def end_time(self) -> float:
        return self._end_time
    
    @property
    def responses(self) -> list[SampleResponse]:
        return self._responses
    
    @property 
    def final_response(self) -> SampleResponse:
        return self._responses[-1]
    
    @property
    def response_time(self) -> float:
        return self.end_time - self.start_time
    
    def __str__(self):
        return f"SampleResult id: {self.sample_id}"
    
    def __repr__(self):
        return str(self)

class ExperimentResults:
    MODEL_NAME = "modelName"
    MODEL_NAME_W_SIZE = "modelNameWSize"
    EXPERIMENT_NAME = "experimentName"
    EXPERIMENT_CODE_NAME = "experimentCodeName"
    MODEL_SIZE = "modelSize"
    MODEL_NORMALIZED_SIZE = "modelNormalizedSize"
    ORACLE_ATTRIBUTES = "hasOracleAttributes"
    ORACLE_SENTENCES = "hasOracleSentences"
    ORACLE_TYPES = "hasOracleTypes"
    ENCODING_METHOD = "encodingMethod"
    ATTRIBUTE_PREDICTION_MODEL = "attributePredictionModel"
    ATTRIBUTE_PREDICTION = "hasAttributePrediction"
    ADDITIONAL_DESCRIPTIONS = "hasAdditionalDescriptions"
    MODEL_FAMILY_COLOR = "modelFamilyColor"
    MODEL_WITH_EXPERIMENT = "modelWithExperiment"
    DATASET_VERSION = "datasetVersion"
    FOLD_NUMBER = "foldNumber"
    def __init__(self, experiment_name: str, result_json: dict):
        self._model_name: str = ""
        self._model_size: float = 0
        self._oracle_attributes: bool = False
        self._oracle_types: bool = False
        self._oracle_sentences: bool = False
        self._additional_description: bool = False
        self._attribute_prediction: bool = False
        self._encoding_method: str = ""
        self._attribute_prediction_model: str = ""
        self._dataset_version: str = ""
        self._fold_number: int = ""
        self._sample_results: list[SampleResult] = self.__parse_results(result_json["results"])
        self.__parse_experiment_name(experiment_name)
        self._experiment_name: str = experiment_name.replace("_ot","")
        self._normalized_size: str = self.__get_normalized_size(self.model_size)

    def __parse_results(self, raw_sample_results: list[dict]) -> list[SampleResult]:
        result_list: list[SampleResult] = []
        for raw_sample_result in raw_sample_results:
            processed_result = SampleResult(raw_sample_result)
            result_list.append(processed_result)
        return result_list

    def __parse_experiment_name(self, experiment_name: str):
        splitted_values = experiment_name.split("_")
        self._model_name = splitted_values[0]
        self._model_size = float(splitted_values[1][:-1])
        self._dataset_version = splitted_values[2]
        self._fold_number = int(splitted_values[3])
        for additional_data in splitted_values[4:]:
            if additional_data.lower() == "ad":
                self._additional_description = True
                continue
            if additional_data.lower() == "oa":
                self._oracle_attributes = True
                continue
            if additional_data.lower() == "ot":
                self._oracle_types = True
                continue
            if additional_data.lower() == "os":
                self._oracle_sentences = True
                continue
            if additional_data.lower() in ("bow", "tfidf", "embeddings"):
                self._encoding_method = additional_data
                continue
            self._attribute_prediction_model = additional_data
            self._attribute_prediction = True 
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_size(self) -> float:
        return self._model_size
    
    @property
    def use_oracle_attributes(self) -> bool:
        return self._oracle_attributes
    
    @property
    def use_oracle_types(self) -> bool:
        return self._oracle_types
    
    @property
    def use_oracle_sentences(self) -> bool:
        return self._oracle_sentences
    
    @property
    def use_additional_descriptions(self) -> bool:
        return self._additional_description
    
    @property
    def sample_results(self) -> list[SampleResult]:
        return self._sample_results
    
    @property
    def experiment_name(self) -> str:
        return self._experiment_name
    
    @property
    def normalized_size(self) -> str:
        return self._normalized_size
    
    @property
    def attribute_prediction_model(self) -> str:
        return self._attribute_prediction_model

    @property
    def attribute_prediction(self) -> str:
        return self._attribute_prediction
    
    @property
    def encoding_method(self) -> str:
        return self._encoding_method
    
    @property
    def dataset_version(self) -> str:
        return self._dataset_version
    
    @property
    def fold_number(self) -> int:
        return self._fold_number

    @property
    def model_name_w_size(self) -> str:
        return f"{self.model_name}:{self.normalized_size}"
    
    @property
    def model_with_experiment_code_name(self) -> str:
        return f"{self.model_name_w_size}_{self.experiment_code_name}"

    def __get_normalized_size(self, size: float) -> str:
        value = int(size) if size == int(size) else size
        return f"{value}b"

    def __str__(self):
        return f"Results of: {self.experiment_name}"
    
    def __repr__(self):
        return str(self)
    
    @property
    def experiment_configuration(self) -> dict:
        return {
            ExperimentResults.MODEL_NAME: self.model_name,
            ExperimentResults.MODEL_NAME_W_SIZE: self.model_name_w_size,
            ExperimentResults.MODEL_NORMALIZED_SIZE: self.normalized_size,
            ExperimentResults.MODEL_WITH_EXPERIMENT: self.model_with_experiment_code_name,
            ExperimentResults.MODEL_SIZE: self.model_size,
            ExperimentResults.EXPERIMENT_NAME: self.experiment_name,
            ExperimentResults.EXPERIMENT_CODE_NAME: self.experiment_code_name,
            ExperimentResults.DATASET_VERSION: self.dataset_version,
            ExperimentResults.FOLD_NUMBER: self.fold_number,
            ExperimentResults.ORACLE_ATTRIBUTES: self.use_oracle_attributes,
            ExperimentResults.ORACLE_SENTENCES: self.use_oracle_sentences,
            ExperimentResults.ORACLE_TYPES: self.use_oracle_types,
            ExperimentResults.ATTRIBUTE_PREDICTION: self.attribute_prediction,
            ExperimentResults.ATTRIBUTE_PREDICTION_MODEL: self.attribute_prediction_model,
            ExperimentResults.ENCODING_METHOD: self.encoding_method,
            ExperimentResults.ADDITIONAL_DESCRIPTIONS: self.use_additional_descriptions,
            ExperimentResults.MODEL_FAMILY_COLOR: "skyblue"
        }
    
    @property
    def experiment_code_name(self) -> str:
        splitted_value = self.experiment_name.split("_", 4)
        if len(splitted_value) < 5:
            return "vanilla"
        return splitted_value[-1].replace("_ot", "")


class ExperimentEvaluator:
    TP = "tp"
    FP = "fp"
    FN = "fn"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    SAMPLE_ID = "sampleID"
    VALID_JSON_RESPONSE = "validJSON"
    NUMBER_OF_RESPONSES = "numberOfResponses"
    TRUE_ATTRIBUTES_COUNT = "trueAttributesCount"
    IDENTIFIED_ATTRIBUTES_COUNT = "identifiedAttributesCount"
    MISSED_ATTRIBUTES_COUNT = "missedAttributesCount"
    CORRECTLY_IDENTIFIED_ATTRIBUTES_COUNT = "correctlyIdentifiedAttributesCount"
    IN_DOMAIN_IDENTIFIED_ATTRIBUTES_COUNT = "inDomainIdentifiedAttributesCount"
    SAMPLE_ATTRIBUTES_STATS = "sampleAttributesStats"
    GLOBAL_ATTRIBUTE_STATS = "globalAttributeStats"
    # MICRO_AVERAGE_METRICS = "microAverageMetrics"
    # MACRO_AVERAGE_METRICS = "macroAverageMetrics"
    METRICS_PER_ATTRIBUTE = "metricsPerAttribute"
    INVALID_OUTPUTS_COUNT = "invalidOutputCount"
    VALID_OUTPUTS_COUNT = "validOutputCount"
    SAMPLE_METRICS = "sampleMetrics"
    RESPONSE_TIME = "responseTime"
    RETRIES = "retries"
    AVERAGE_RESPONSE_TIME = "averageResponseTime"
    AVERAGE_RETRIES = "averageRetries"
    def __init__(self, rave_dataset: list[RaveSample], valid_attributes: list[str]):
        self._rave_dataset = rave_dataset
        self._rave_sample_id_mapping = {sample.id: sample for sample in rave_dataset }
        self._all_valid_attributes_set = set(valid_attributes)

    def evaluate_experiment(self, experiment: ExperimentResults) -> dict:
        raw_result_pair_dict = self.make_raw_and_result_pair(experiment)
        sample_results_stats_list: list[dict] = []
        for _, raw_result_pair in raw_result_pair_dict.items():
            rave_sample, experiment_result = raw_result_pair
            predicted_values = experiment_result.final_response.parsed_json
            true_values = rave_sample.text_attributes

            sample_result_stats = self.handle_base_sample_stats(experiment_result)
            sample_result_stats |= self.handle_base_attribute_stats(experiment_result, rave_sample)
            sample_result_stats[ExperimentEvaluator.SAMPLE_ATTRIBUTES_STATS] = self.calculate_sample_tp_fp_fn_for_all_attributes(predicted_values, true_values)
            sample_result_stats[ExperimentEvaluator.SAMPLE_METRICS] = self.calculate_sample_metrics(sample_result_stats[ExperimentEvaluator.SAMPLE_ATTRIBUTES_STATS])

            sample_results_stats_list.append(sample_result_stats)
        return self.calculate_experiment_global_stats(sample_results_stats_list) | {
            "sampleStats": sample_results_stats_list
        }
    
    def calculate_experiment_global_stats(self, sample_results_stats_list: list[dict]) -> dict:
        global_attribute_stats: dict[str, dict[str, int]] = self.__get_global_attribute_stats(sample_results_stats_list)
        attributes_p_r_f1_dict: dict[str, dict[str, float]] = {}
        
        for attr_name, stats_dict in global_attribute_stats.items():
            attributes_p_r_f1_dict[attr_name] = self.evaluate_metrics(stats_dict)

        total_samples: int = len(sample_results_stats_list)
        total_response_time: float = 0
        total_retries: int = 0
        total_valid_outputs: int = 0
        
        for sample_stats in sample_results_stats_list:
            total_response_time += sample_stats[ExperimentEvaluator.RESPONSE_TIME]
            total_retries += sample_stats[ExperimentEvaluator.RETRIES]
            total_valid_outputs += 1 if sample_stats[ExperimentEvaluator.VALID_JSON_RESPONSE] else 0
                
        return self.calculate_micro_average(global_attribute_stats) | self.calculate_macro_average(attributes_p_r_f1_dict) | {
            ExperimentEvaluator.VALID_OUTPUTS_COUNT: total_valid_outputs,
            ExperimentEvaluator.INVALID_OUTPUTS_COUNT: total_samples - total_valid_outputs,
            ExperimentEvaluator.AVERAGE_RETRIES: total_retries / total_samples,
            ExperimentEvaluator.AVERAGE_RESPONSE_TIME: total_response_time / total_samples,
            ExperimentEvaluator.GLOBAL_ATTRIBUTE_STATS : global_attribute_stats,
            ExperimentEvaluator.METRICS_PER_ATTRIBUTE : attributes_p_r_f1_dict,
        }
    
    def __get_global_attribute_stats(self, sample_results_stats_list: list[dict[str, dict[str, dict]]]) -> dict:
        global_attribute_stats: dict[str, dict] = defaultdict(lambda: {ExperimentEvaluator.TP: 0, ExperimentEvaluator.FP: 0, ExperimentEvaluator.FN: 0})
        for sample_result in sample_results_stats_list:
            for attr_name, attr_stats in sample_result[ExperimentEvaluator.SAMPLE_ATTRIBUTES_STATS].items():
                global_attr_stats = global_attribute_stats[attr_name]
                for stat_name, stat_value in attr_stats.items():
                    global_attr_stats[stat_name] += stat_value
        return dict(global_attribute_stats)

    def calculate_sample_tp_fp_fn_for_all_attributes(self, predicted_attrs: dict, gold_attrs: dict) -> dict[str, dict[str, int]]:
        attribute_metrics: dict[str, dict] = defaultdict(lambda: {ExperimentEvaluator.TP: 0, ExperimentEvaluator.FP: 0, ExperimentEvaluator.FN: 0})
        all_attributes = self.get_not_null_keys(predicted_attrs) | set(gold_attrs.keys())
        for attr_name in all_attributes:
            stats = attribute_metrics[attr_name]
            category_to_add = self.resolve_attr_category(attr_name, predicted_attrs, gold_attrs)
            stats[category_to_add] += 1
        return dict(attribute_metrics)

    def handle_base_sample_stats(self, sample_result: SampleResult) -> dict:
        return {
            ExperimentEvaluator.SAMPLE_ID: sample_result.sample_id,
            ExperimentEvaluator.VALID_JSON_RESPONSE: sample_result.final_response.has_valid_json,
            ExperimentEvaluator.NUMBER_OF_RESPONSES: len(sample_result.responses),
            ExperimentEvaluator.RESPONSE_TIME: sample_result.response_time,
            ExperimentEvaluator.RETRIES: len(sample_result.responses) - 1
        }

    def handle_base_attribute_stats(self, sample_result: SampleResult, rave_sample: RaveSample) -> dict:
        rave_correct_attributes = set(rave_sample.text_attributes.keys())
        predicted_not_null_attributes = self.get_not_null_keys(sample_result.final_response.parsed_json)
        return {
            ExperimentEvaluator.TRUE_ATTRIBUTES_COUNT: len(rave_correct_attributes),
            ExperimentEvaluator.IDENTIFIED_ATTRIBUTES_COUNT: len(predicted_not_null_attributes),
            ExperimentEvaluator.CORRECTLY_IDENTIFIED_ATTRIBUTES_COUNT: len(rave_correct_attributes & predicted_not_null_attributes),
            ExperimentEvaluator.IN_DOMAIN_IDENTIFIED_ATTRIBUTES_COUNT: len(self._all_valid_attributes_set & predicted_not_null_attributes),
            ExperimentEvaluator.MISSED_ATTRIBUTES_COUNT: len(rave_correct_attributes - predicted_not_null_attributes)
        }
    
    def get_not_null_keys(self, dict_to_handle: dict) -> set:
        return {key for key, value in dict_to_handle.items() if value is not None}

    def make_raw_and_result_pair(self, experiment_result: ExperimentResults) -> dict[int, tuple[RaveSample, SampleResult]]:
        sample_result_id_map = { sample.sample_id : sample for sample in experiment_result.sample_results }
        result_dict: dict[int, tuple[RaveSample, SampleResult]] = {}
        for sample_id, sample_result in sample_result_id_map.items():
            rave_sample = self._rave_sample_id_mapping[sample_id]
            result_dict[sample_id] = (rave_sample, sample_result)
        return result_dict
    
    def resolve_attr_category(self, attr_name: str, predicted_attrs:dict, gold_attrs: dict) -> str:
        if attr_name in predicted_attrs and attr_name in gold_attrs:
            return ExperimentEvaluator.TP if predicted_attrs[attr_name] == gold_attrs[attr_name] else ExperimentEvaluator.FP
        if attr_name in predicted_attrs and attr_name not in gold_attrs:
            return ExperimentEvaluator.FP
        return ExperimentEvaluator.FN
    
    def calculate_precision(self, metrics_dict: dict) -> float:
        tp = metrics_dict[ExperimentEvaluator.TP]
        fp = metrics_dict[ExperimentEvaluator.FP]
        return (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0

    def calculate_recall(self, metrics_dict: dict) -> float:
        tp = metrics_dict[ExperimentEvaluator.TP]
        fn = metrics_dict[ExperimentEvaluator.FN]
        return (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0

    def calculate_f1(self, metric_dict: dict):
        precision = self.calculate_precision(metric_dict)
        recall = self.calculate_recall(metric_dict)
        return ((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    
    def calculate_sample_metrics(self, attributes_tp_fp_fn_dict: dict[str, dict[str, int]]) -> dict[str, float]:
        overall_metric = {
            ExperimentEvaluator.TP: 0,
            ExperimentEvaluator.FP: 0,
            ExperimentEvaluator.FN: 0
        }
        for _, attr_metrics in attributes_tp_fp_fn_dict.items():
            for metric_key, metric_value in attr_metrics.items():
                overall_metric[metric_key] += metric_value
        return self.evaluate_metrics(overall_metric)


    def calculate_micro_average(self, attributes_tp_fp_fn_dict: dict[str, dict[str, int]]) -> dict[str, float]:
        overall_metric = {
            ExperimentEvaluator.TP: 0,
            ExperimentEvaluator.FP: 0,
            ExperimentEvaluator.FN: 0
        }
        for _, attr_metrics in attributes_tp_fp_fn_dict.items():
            for metric_key, metric_value in attr_metrics.items():
                overall_metric[metric_key] += metric_value
        
        evaluated_metrics = {f"micro_{key}":value for key, value in self.evaluate_metrics(overall_metric).items()}
        return evaluated_metrics | {
            f"{ExperimentEvaluator.TP}_total": overall_metric[ExperimentEvaluator.TP],
            f"{ExperimentEvaluator.FP}_total": overall_metric[ExperimentEvaluator.FP],
            f"{ExperimentEvaluator.FN}_total": overall_metric[ExperimentEvaluator.FN],
        }

    def calculate_macro_average(self, attributes_p_r_f1_dict: dict[str, dict]) -> dict:
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0

        for _, metrics in attributes_p_r_f1_dict.items():
            precision_sum += metrics[ExperimentEvaluator.PRECISION]
            recall_sum += metrics[ExperimentEvaluator.RECALL]
            f1_sum += metrics[ExperimentEvaluator.F1_SCORE]
        number_of_attributes = len(attributes_p_r_f1_dict)
        return {
            f"macro_{ExperimentEvaluator.PRECISION}": precision_sum / number_of_attributes,
            f"macro_{ExperimentEvaluator.RECALL}": recall_sum / number_of_attributes,
            f"macro_{ExperimentEvaluator.F1_SCORE}": f1_sum / number_of_attributes,
        }
    
    def evaluate_metrics(self, metric_dict: dict) -> dict:
        return {
            ExperimentEvaluator.PRECISION : self.calculate_precision(metric_dict),
            ExperimentEvaluator.RECALL : self.calculate_recall(metric_dict),
            ExperimentEvaluator.F1_SCORE : self.calculate_f1(metric_dict)
        }