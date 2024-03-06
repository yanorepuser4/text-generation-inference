use std::collections::BTreeMap;

/// Payload validation logic
use crate::validation::ValidationError::{BestOfSampling, BestOfSeed, EmptyInput};
use crate::{GenerateParameters, GenerateRequest, GrammarType};
use jsonschema::{Draft, JSONSchema};
use rand::{thread_rng, Rng};
use serde_json::Value;
use text_generation_client::{
    GrammarType as ProtoGrammarType, NextTokenChooserParameters, StatesToTokenMaps,
    StoppingCriteriaParameters,
};
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::TruncationDirection;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{instrument, Span};

use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Validation parameters
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,
    max_total_tokens: usize,
    disable_grammar_support: bool,
    /// Channel to communicate with the background tokenization task
    sender: Option<mpsc::UnboundedSender<TokenizerRequest>>,
    grammar_compilation_sender: Option<mpsc::UnboundedSender<GrammarCompilationRequest>>,
}

impl Validation {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        workers: usize,
        tokenizer: Option<Tokenizer>,
        max_best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_length: usize,
        max_total_tokens: usize,
        disable_grammar_support: bool,
    ) -> Self {
        // If we have a fast tokenizer
        let (sender, grammar_compilation_sender) = if let Some(tokenizer) = tokenizer {
            // Create round robin channel
            let (validation_sender, validation_round_robin_receiver) = mpsc::unbounded_channel();
            let mut senders = Vec::with_capacity(workers);

            // Create workers
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
                senders.push(tokenizer_sender);

                // Spawn worker
                tokio::task::spawn_blocking(move || {
                    tokenizer_worker(tokenizer_clone, tokenizer_receiver)
                });
            }

            // Create round robin channel
            let (grammar_sender, grammar_round_robin_receiver) = mpsc::unbounded_channel();
            let mut grammar_senders = Vec::with_capacity(workers);

            // Create workers
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let (grammar_sender, grammar_receiver) = mpsc::unbounded_channel();
                grammar_senders.push(grammar_sender);

                // Spawn worker
                tokio::task::spawn_blocking(move || {
                    grammar_compilation_worker(tokenizer_clone, grammar_receiver).map_err(|e| {
                        tracing::error!("Error in grammar compilation worker: {:?}", e);
                        e
                    })
                });
            }

            // Create tokenization round robin task
            tokio::spawn(round_robin_task::<TokenizerRequest>(
                validation_round_robin_receiver,
                senders,
            ));

            // Create grammar compilation round robin task
            tokio::spawn(round_robin_task::<GrammarCompilationRequest>(
                grammar_round_robin_receiver,
                grammar_senders,
            ));

            (Some(validation_sender), Some(grammar_sender))
        } else {
            (None, None)
        };

        Self {
            max_best_of,
            sender,
            grammar_compilation_sender,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        }
    }

    #[instrument(skip(self, inputs))]
    pub async fn tokenize(
        &self,
        inputs: String,
        truncate: Option<usize>,
    ) -> Result<Option<(tokenizers::Encoding, String)>, ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender
                .send(((inputs, truncate), response_sender, Span::current()))
                .unwrap();

            // Await on response channel
            // Unwrap is safe here
            let encoding = response_receiver.await.unwrap()?;
            Ok(Some(encoding))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self, inputs))]
    pub async fn compile_grammar(
        &self,
        inputs: String,
    ) -> Result<(String, StateTokenMaps), ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.grammar_compilation_sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender
                .send((inputs.clone(), response_sender, Span::current()))
                .unwrap();

            // Await on response channel
            // Unwrap is safe here
            let encoding = response_receiver.await.unwrap()?;
            return Ok(encoding);
        }

        Ok((String::new(), BTreeMap::new()))
    }

    #[instrument(skip(self, inputs))]
    async fn validate_input(
        &self,
        inputs: String,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
    ) -> Result<(String, usize, u32), ValidationError> {
        // If we have a fast tokenizer
        if let Some((encoding, inputs)) = self.tokenize(inputs.clone(), truncate).await? {
            // Create response channel
            let input_length = encoding.len();

            // Get total tokens
            let max_new_tokens: u32 = if let Some(max_new_tokens) = max_new_tokens {
                max_new_tokens
            } else {
                self.max_total_tokens.saturating_sub(input_length) as u32
            };
            let total_tokens = input_length + max_new_tokens as usize;

            // Validate MaxTotalTokens
            if total_tokens > self.max_total_tokens {
                return Err(ValidationError::MaxTotalTokens(
                    self.max_total_tokens,
                    input_length,
                    max_new_tokens,
                ));
            }

            // Validate InputLength
            if input_length > self.max_input_length {
                return Err(ValidationError::InputLength(
                    self.max_input_length,
                    input_length,
                ));
            }

            metrics::histogram!("tgi_request_input_length", input_length as f64);
            Ok((inputs, input_length, max_new_tokens))
        }
        // Return inputs without validation
        else {
            // In this case, we don't know the real length in tokens of the inputs
            // However, the inputs will be truncated by the python servers
            // We make sure that truncate + max_new_tokens <= self.max_total_tokens
            let max_new_tokens: u32 = if let Some(max_new_tokens) = max_new_tokens {
                max_new_tokens
            } else if let Some(truncate) = truncate {
                self.max_total_tokens.saturating_sub(truncate) as u32
            } else {
                return Err(ValidationError::UnsetMaxNewTokens);
            };
            let input_length = truncate.unwrap_or(self.max_input_length);

            // Validate MaxNewTokens
            if (input_length as u32 + max_new_tokens) > self.max_total_tokens as u32 {
                return Err(ValidationError::MaxNewTokens(
                    self.max_total_tokens - self.max_input_length,
                    max_new_tokens,
                ));
            }

            Ok((inputs, input_length, max_new_tokens))
        }
    }

    /// Validate a payload and get the number of tokens in the input
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        let GenerateParameters {
            best_of,
            temperature,
            repetition_penalty,
            frequency_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            max_new_tokens,
            stop: stop_sequences,
            truncate,
            seed,
            watermark,
            decoder_input_details,
            top_n_tokens,
            grammar,
            ..
        } = request.parameters;

        // sampling must be true when best_of > 1
        let best_of = best_of.unwrap_or(1);
        let sampling = do_sample
            || temperature.is_some()
            || top_k.is_some()
            || top_p.is_some()
            || typical_p.is_some();

        if best_of > 1 && !sampling {
            return Err(BestOfSampling);
        }

        let temperature = temperature.unwrap_or(1.0);
        if temperature <= 0.0 {
            return Err(ValidationError::Temperature);
        }

        let repetition_penalty = repetition_penalty.unwrap_or(1.0);
        if repetition_penalty <= 0.0 {
            return Err(ValidationError::RepetitionPenalty);
        }

        let frequency_penalty = frequency_penalty.unwrap_or(0.0);
        if !(-2.0..=2.0).contains(&frequency_penalty) {
            return Err(ValidationError::FrequencyPenalty);
        }

        // Different because the proto default value is not a valid value
        // for the user
        let top_p = top_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TopP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let typical_p = typical_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TypicalP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let top_k: u32 = top_k
            .map(|value| {
                if value <= 0 {
                    return Err(ValidationError::TopK);
                }
                Ok(value as u32)
            })
            .unwrap_or(Ok(0))?;

        if max_new_tokens == Some(0) {
            return Err(ValidationError::NegativeMaxNewTokens);
        }

        if stop_sequences.len() > self.max_stop_sequences {
            return Err(ValidationError::StopSequence(
                self.max_stop_sequences,
                stop_sequences.len(),
            ));
        }

        // If seed is None, assign a random one
        let seed = match seed {
            None => thread_rng().gen(),
            Some(seed) => {
                if best_of > 1 {
                    return Err(BestOfSeed);
                }
                seed
            }
        };

        let top_n_tokens = top_n_tokens
            .map(|value| {
                if value > self.max_top_n_tokens {
                    return Err(ValidationError::TopNTokens(self.max_top_n_tokens, value));
                }
                Ok(value)
            })
            .unwrap_or(Ok(0))?;

        // Check if inputs is empty
        if request.inputs.is_empty() {
            return Err(EmptyInput);
        }

        // Check if truncate is strictly positive and less than max_input_length
        let truncate = truncate
            .map(|value| {
                if value == 0 || value > self.max_input_length {
                    return Err(ValidationError::Truncate(self.max_input_length, value));
                }
                Ok(Some(value))
            })
            .unwrap_or(Ok(None))?;

        // Validate inputs
        let (inputs, input_length, max_new_tokens) = self
            .validate_input(request.inputs, truncate, max_new_tokens)
            .await?;

        // TODO: we should build the FSM here and pass the compiled FSM instead of the grammar
        // NOTE: this is currently difficult because we need the tokenizer in Python to build
        // the FSM and we'd have to load a copy of the tokenizer into our Pyo3 instance which
        // may be slow and memory intensive. Best case is to have a Rust implementation of the FSM
        // compiler and use that to build the FSM here.

        // Validate grammar and unpack the grammar and type for the proto message
        let (grammar, grammar_type, states_to_token_maps) = match grammar {
            Some(grammar) => {
                // Ensure that grammar is not set if it's not supported
                if self.disable_grammar_support {
                    return Err(ValidationError::Grammar);
                }
                match grammar {
                    GrammarType::Json(json) => {
                        let json = match json {
                            // if value is a string, we need to parse it again to make sure its
                            // a valid json
                            Value::String(s) => serde_json::from_str(&s)
                                .map_err(|e| ValidationError::InvalidGrammar(e.to_string())),
                            Value::Object(_) => Ok(json),
                            _ => Err(ValidationError::Grammar),
                        }?;

                        // Check if the json is a valid JSONSchema
                        JSONSchema::options()
                            .with_draft(Draft::Draft202012)
                            .compile(&json)
                            .map_err(|e| ValidationError::InvalidGrammar(e.to_string()))?;

                        // NOTE: this is the first step to compile the grammar
                        let (regex_compiled_grammar, _states_to_token_maps) = self
                            .compile_grammar(serde_json::to_string(&json).unwrap())
                            .await
                            .map_err(|e| ValidationError::InvalidGrammar(e.to_string()))?;

                        let stm = StatesToTokenMaps {
                            start_states: vec![],
                            tokens: vec![],
                            end_states: vec![],
                        };

                        (
                            regex_compiled_grammar,
                            ProtoGrammarType::Regex.into(),
                            Some(stm),
                        )
                    }
                    GrammarType::Regex(regex) => (regex, ProtoGrammarType::Regex.into(), None),
                }
            }
            None => (String::new(), ProtoGrammarType::None.into(), None),
        };

        let parameters = NextTokenChooserParameters {
            temperature,
            repetition_penalty,
            frequency_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            seed,
            watermark,
            grammar,
            grammar_type,
            states_to_token_maps,
        };
        let stopping_parameters = StoppingCriteriaParameters {
            max_new_tokens,
            stop_sequences,
            ignore_eos_token: false,
        };

        metrics::histogram!("tgi_request_max_new_tokens", max_new_tokens as f64);

        Ok(ValidGenerateRequest {
            inputs,
            decoder_input_details,
            input_length: input_length as u32,
            truncate: truncate.unwrap_or(self.max_input_length) as u32,
            parameters,
            stopping_parameters,
            top_n_tokens,
        })
    }

    /// Validate the best_of parameter
    #[instrument(skip_all)]
    pub(crate) fn validate_best_of(&self, best_of: usize) -> Result<usize, ValidationError> {
        if self.max_best_of == 1 && best_of != 1 {
            return Err(ValidationError::BestOfDisabled);
        }

        if best_of > self.max_best_of {
            return Err(ValidationError::BestOf(self.max_best_of, best_of));
        }

        Ok(best_of)
    }
}

/// Round robin tokenization task
async fn round_robin_task<T>(
    mut receiver: mpsc::UnboundedReceiver<T>,
    senders: Vec<mpsc::UnboundedSender<T>>,
) {
    loop {
        for sender in &senders {
            match receiver.recv().await {
                None => return,
                Some(request) => sender.send(request).unwrap(),
            };
        }
    }
}

/// Start tokenization workers
fn tokenizer_worker(tokenizer: Tokenizer, mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>) {
    // Loop over requests
    while let Some(((inputs, truncate), response_tx, parent_span)) = receiver.blocking_recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(prepare_input(inputs, truncate, &tokenizer))
                .unwrap_or(())
        })
    }
}

/// Start grammar compilation workers
fn grammar_compilation_worker(
    tokenizer: Tokenizer,
    mut receiver: mpsc::UnboundedReceiver<GrammarCompilationRequest>,
) -> Result<(), PyErr> {
    // initialize python runtime
    pyo3::prepare_freethreaded_python();

    // load in outlines for all workers
    Python::with_gil(|py| {
        PyModule::import(py, "outlines")?;
        Ok::<_, PyErr>(())
    })?;

    // Loop over requests
    while let Some((inputs, response_tx, parent_span)) = receiver.blocking_recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(compile_grammar(inputs, &tokenizer))
                .unwrap_or(())
        })
    }

    Ok(())
}

/// Get input length and optionally truncate it
fn prepare_input(
    mut inputs: String,
    truncate: Option<usize>,
    tokenizer: &Tokenizer,
) -> Result<(tokenizers::Encoding, String), ValidationError> {
    // Get the number of tokens in the input
    let mut encoding = tokenizer
        .encode(inputs.clone(), true)
        .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;

    // Optionally truncate
    if let Some(truncate) = truncate {
        if truncate < encoding.len() {
            encoding.truncate(truncate, 0, TruncationDirection::Left);
            inputs = tokenizer
                .decode(encoding.get_ids(), false)
                .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;
        }
    }

    Ok((encoding, inputs))
}

type StateTokenMaps = BTreeMap<u32, BTreeMap<u32, u32>>;

/// Compile a grammar
fn compile_grammar(
    inputs: String,
    tokenizer: &Tokenizer,
) -> Result<(String, StateTokenMaps), ValidationError> {
    let start_time = std::time::Instant::now();
    let (schema, states_to_token_maps) = Python::with_gil(|py| -> PyResult<(_, _)> {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            r#"
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
import time
from transformers.file_utils import SPIECE_UNDERLINE

class Tokenizer:
    def __init__(self, vocab, special_tokens):
        self.vocabulary = vocab
        self.special_tokens = special_tokens
        self.eos_token_id = 0

    def get_vocab(self, with_added_tokens):
        return self.vocabulary

    def encode(self, text, add_special_tokens):
        return text

    def decode(self, text, skip_special_tokens):
        return text

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

def adapt_tokenizer(vocab, special_tokens):
    start_time = time.time()
    tokenizer = Tokenizer(vocab, special_tokens)

    def convert_token_to_string(token: str) -> str:

        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
            return " " + string

        return string

    tokenizer.convert_token_to_string = convert_token_to_string
    print(f"Adapted tokenizer in {time.time() - start_time:.2f}s")
    return tokenizer

def compile_regex_grammar(inputs, vocab, special_tokens):
    start_time = time.time()
    print("🔥 starting compile_regex_grammar", inputs)
    schema = build_regex_from_schema(inputs)
    print(f"Compiled grammar in {time.time() - start_time:.2f}s")
    tokenizer = adapt_tokenizer(vocab, special_tokens)
    print(f"Adapted tokenizer in {time.time() - start_time:.2f}s")
    fsm = RegexFSM(schema, tokenizer)
    print(f"Compiled grammar in {time.time() - start_time:.2f}s")
    return fsm

def convert_grammar_to_regex(inputs):
    start_time = time.time()
    print("🔥 starting convert_grammar_to_regex", inputs)
    schema = build_regex_from_schema(inputs)
    print(f"Compiled grammar in {time.time() - start_time:.2f}s")
    return schema
"#,
            "",
            "",
        )?
        .into_py(py);

        let convert_grammar_to_regex = fun.getattr(py, "convert_grammar_to_regex")?;
        let compile_regex_grammar = fun.getattr(py, "compile_regex_grammar")?;

        let args: &pyo3::types::PyDict = tokenizer.get_vocab(true).into_py_dict(py);
        let special_tokens: Vec<String> = vec![];

        let regex_fsm = convert_grammar_to_regex.call(py, (inputs.clone(),), None)?;

        let compiled_grammar =
            compile_regex_grammar.call(py, (inputs.clone(), args, special_tokens), None)?;
        let compiled_grammar_ref = compiled_grammar.into_ref(py);

        let states_to_token_maps = compiled_grammar_ref
            .getattr("states_to_token_maps")?
            .extract::<StateTokenMaps>()?;

        println!("🔥 elapsed: {:?}", start_time.elapsed());

        // size of serialized states_to_token_maps
        let states_to_token_maps_json = serde_json::to_string(&states_to_token_maps).unwrap();
        println!(
            "🔥 states_to_token_maps size: {:.2}MB",
            states_to_token_maps_json.len() as f64 / 1024.0 / 1024.0
        );

        let result = regex_fsm.into_ref(py).extract().unwrap();

        println!("result: {:?}", result);

        Ok((result, states_to_token_maps))
    })
    .map_err(|e| ValidationError::InvalidGrammar(e.to_string()))?;
    let elapsed = start_time.elapsed();
    println!("🔥 elapsed: {:?}", elapsed);
    Ok((schema, states_to_token_maps))
}

type GrammarCompilationRequest = (
    String,
    oneshot::Sender<Result<(String, StateTokenMaps), ValidationError>>,
    Span,
);

type TokenizerRequest = (
    (String, Option<usize>),
    oneshot::Sender<Result<(tokenizers::Encoding, String), ValidationError>>,
    Span,
);

#[derive(Debug, Clone)]
pub(crate) struct ValidGenerateRequest {
    pub inputs: String,
    pub input_length: u32,
    pub truncate: u32,
    pub decoder_input_details: bool,
    pub parameters: NextTokenChooserParameters,
    pub stopping_parameters: StoppingCriteriaParameters,
    pub top_n_tokens: u32,
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("`best_of` must be > 0 and <= {0}. Given: {1}")]
    BestOf(usize, usize),
    #[error("`best_of` != 1 is not allowed for this endpoint")]
    BestOfDisabled,
    #[error("you must use sampling when `best_of` is > 1")]
    BestOfSampling,
    #[error("`seed` must not be set when `best_of` > 1")]
    BestOfSeed,
    #[error("`best_of` != 1 is not supported when streaming tokens")]
    BestOfStream,
    #[error("`top_n_tokens` must be >= 0 and <= {0}. Given: {1}")]
    TopNTokens(u32, u32),
    #[error("`top_n_tokens` != 0 is not allowed for this endpoint")]
    TopNTokensDisabled,
    #[error("`decoder_input_details` == true is not supported when streaming tokens")]
    PrefillDetailsStream,
    #[error("`temperature` must be strictly positive")]
    Temperature,
    #[error("`repetition_penalty` must be strictly positive")]
    RepetitionPenalty,
    #[error("`frequency_penalty` must be >= -2.0 and <= 2.0")]
    FrequencyPenalty,
    #[error("`top_p` must be > 0.0 and < 1.0")]
    TopP,
    #[error("`top_k` must be strictly positive")]
    TopK,
    #[error("`truncate` must be strictly positive and less than {0}. Given: {1}")]
    Truncate(usize, usize),
    #[error("`typical_p` must be > 0.0 and < 1.0")]
    TypicalP,
    #[error("one of `max_new_tokens` or `truncate` must be set if a fast tokenizer is not in use")]
    UnsetMaxNewTokens,
    #[error("`max_new_tokens` must be strictly positive")]
    NegativeMaxNewTokens,
    #[error("`max_new_tokens` must be <= {0}. Given: {1}")]
    MaxNewTokens(usize, u32),
    #[error("`inputs` tokens + `max_new_tokens` must be <= {0}. Given: {1} `inputs` tokens and {2} `max_new_tokens`")]
    MaxTotalTokens(usize, usize, u32),
    #[error("`inputs` must have less than {0} tokens. Given: {1}")]
    InputLength(usize, usize),
    #[error("`inputs` cannot be empty")]
    EmptyInput,
    #[error("`stop` supports up to {0} stop sequences. Given: {1}")]
    StopSequence(usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
    #[error("grammar is not supported")]
    Grammar,
    #[error("grammar is not valid: {0}")]
    InvalidGrammar(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_parameters;
    use crate::tests::get_tokenizer;

    #[tokio::test]
    async fn test_validation_max_new_tokens() {
        let tokenizer = None;
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let workers = 1;
        let disable_grammar_support = true;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );

        let max_new_tokens = 10;
        match validation
            .validate_input("Hello".to_string(), None, Some(max_new_tokens))
            .await
        {
            Err(ValidationError::MaxNewTokens(1, 10)) => (),
            _ => panic!("Unexpected not max new tokens"),
        }
    }

    #[tokio::test]
    async fn test_validation_input_length() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let disable_grammar_support = true;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );

        let max_new_tokens = 10;
        match validation
            .validate_input("Hello".to_string(), None, Some(max_new_tokens))
            .await
        {
            Err(ValidationError::MaxTotalTokens(6, 1, 10)) => (),
            _ => panic!("Unexpected not max new tokens"),
        }
    }

    #[tokio::test]
    async fn test_validation_best_of_sampling() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let workers = 1;
        let disable_grammar_support = true;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    best_of: Some(2),
                    do_sample: false,
                    ..default_parameters()
                },
            })
            .await
        {
            Err(ValidationError::BestOfSampling) => (),
            _ => panic!("Unexpected not best of sampling"),
        }
    }

    #[tokio::test]
    async fn test_validation_top_p() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 106;
        let workers = 1;
        let disable_grammar_support = true;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_p: Some(1.0),
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
        {
            Err(ValidationError::TopP) => (),
            _ => panic!("Unexpected top_p"),
        }

        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_p: Some(0.99),
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
        {
            Ok(_) => (),
            _ => panic!("Unexpected top_p error"),
        }

        let valid_request = validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_p: None,
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
            .unwrap();
        // top_p == 1.0 is invalid for users to ask for but it's the default resolved value.
        assert_eq!(valid_request.parameters.top_p, 1.0);
    }

    #[tokio::test]
    async fn test_validation_top_n_tokens() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequences = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 106;
        let workers = 1;
        let disable_grammar_support = true;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: Some(5),
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
        {
            Err(ValidationError::TopNTokens(4, 5)) => (),
            _ => panic!("Unexpected top_n_tokens"),
        }

        validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: Some(4),
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
            .unwrap();

        validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: Some(0),
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
            .unwrap();

        let valid_request = validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: None,
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
            .unwrap();

        assert_eq!(valid_request.top_n_tokens, 0);
    }
}
