package openai

import (
	"context"
	"fmt"
	"iter"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type openaiModel struct {
	name   string
	client *openai.Client
}

func NewModel(ctx context.Context, modelName string, opts ...option.RequestOption) (model.LLM, error) {
	client := openai.NewClient(opts...)

	return &openaiModel{
		name:   modelName,
		client: &client,
	}, nil
}

func (o *openaiModel) Name() string {
	return o.name
}

func (o *openaiModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	o.maybeAppendUserContent(req)

	body, err := LLMRequest2ChatCompletionNewParams(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	if stream {
		return o.generateStream(ctx, body)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := o.generate(ctx, body)
		yield(resp, err)
	}
}

func (o *openaiModel) generate(ctx context.Context, body *openai.ChatCompletionNewParams) (*model.LLMResponse, error) {
	chatCompletion, err := o.client.Chat.Completions.New(ctx, *body)
	if err != nil {
		return nil, fmt.Errorf("failed to generate content: %w", err)
	}
	resp := ChatCompletion2LLMResponse(chatCompletion)
	return resp, nil
}

func (o *openaiModel) generateStream(ctx context.Context, body *openai.ChatCompletionNewParams) iter.Seq2[*model.LLMResponse, error] {
	body.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: param.NewOpt(true),
	}

	stream := o.client.Chat.Completions.NewStreaming(ctx, *body)

	return func(yield func(*model.LLMResponse, error) bool) {
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()
			resp := convertChunk(chunk)
			if resp != nil {
				if !yield(resp, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("failed to generate stream content: %w", err))
		}
	}
}

func (o *openaiModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

func LLMRequest2ChatCompletionNewParams(req *model.LLMRequest) (*openai.ChatCompletionNewParams, error) {
	params := &openai.ChatCompletionNewParams{
		Model: shared.ChatModel(req.Model),
	}
	if err := applyGenerationConfig(params, req.Config); err != nil {
		return nil, err
	}

	contents := covertContents(req.Contents)
	params.Messages = append(params.Messages, contents...)
	return params, nil
}

func covertContents(contents []*genai.Content) []openai.ChatCompletionMessageParamUnion {
	var (
		messages  []openai.ChatCompletionMessageParamUnion
		texts     []string
		curRole   genai.Role
		flushText = func() {
			if len(texts) == 0 {
				return
			}
			msg := newMessages(curRole, texts)
			if msg == nil {
				return
			}
			messages = append(messages, msg...)
			texts = texts[:0]
		}
	)

	for _, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		curRole = genai.Role(content.Role)
		for _, part := range content.Parts {
			switch {
			case part == nil:
				continue
			case part.Text != "":
				texts = append(texts, part.Text)
			}
		}
		flushText()

	}

	return messages
}

func covertSystemMessage(systemInstruction *genai.Content) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	if systemInstruction == nil || len(systemInstruction.Parts) == 0 {
		return nil
	}

	for _, part := range systemInstruction.Parts {
		switch {
		case part == nil:
			continue
		case part.Text != "":
			messages = append(messages, openai.SystemMessage(part.Text))
		}
	}

	return messages
}

func newMessages(role genai.Role, texts []string) []openai.ChatCompletionMessageParamUnion {
	var msgFunc func(content string) openai.ChatCompletionMessageParamUnion

	switch role {
	case "", genai.RoleUser:
		msgFunc = openai.UserMessage
	case genai.RoleModel:
		msgFunc = openai.AssistantMessage
	case "system":
		msgFunc = openai.SystemMessage
	case "developer":
		msgFunc = openai.DeveloperMessage
	default:
		return nil
	}

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(texts))
	for _, text := range texts {
		messages = append(messages, msgFunc(text))
	}
	return messages
}

func ChatCompletion2LLMResponse(resp *openai.ChatCompletion) *model.LLMResponse {
	if resp == nil {
		return nil
	}
	usageMetadata := convertUsage(resp.Usage)
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			UsageMetadata: usageMetadata,
			ErrorCode:     "UNKNOWN_ERROR",
			ErrorMessage:  "Unknown error.",
		}
	}

	choice := resp.Choices[0]
	message := choice.Message
	content := &genai.Content{
		Role: genai.RoleModel,
	}
	if message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: message.Content})
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usageMetadata,
		FinishReason:  finishReason(choice.FinishReason),
	}
}

func convertChunk(chunk openai.ChatCompletionChunk) *model.LLMResponse {
	if len(chunk.Choices) == 0 {
		if chunk.JSON.Usage.Valid() {
			return &model.LLMResponse{
				UsageMetadata: convertUsage(chunk.Usage),
				TurnComplete:  true,
			}
		}
		return nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	content := &genai.Content{
		Role: genai.RoleModel,
	}

	if delta.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: delta.Content})
	}

	// TODO: 阶段3 - 处理 delta.ToolCalls (增量 function call)

	resp := &model.LLMResponse{
		Content: content,
		Partial: true,
	}

	// 检查是否是最后一个 choice chunk
	if choice.FinishReason != "" {
		resp.TurnComplete = true
		resp.Partial = false
		resp.FinishReason = finishReason(choice.FinishReason)
		if chunk.JSON.Usage.Valid() { // ← 添加检查
			resp.UsageMetadata = convertUsage(chunk.Usage)
		}
	}

	return resp
}

func convertUsage(usage openai.CompletionUsage) *genai.GenerateContentResponseUsageMetadata {
	metadata := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.PromptTokens),
		CandidatesTokenCount: int32(usage.CompletionTokens),
		TotalTokenCount:      int32(usage.TotalTokens),
		ThoughtsTokenCount:   int32(usage.CompletionTokensDetails.ReasoningTokens),
	}

	return metadata
}

func finishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls":
		return genai.FinishReasonStop
	case "content_filter":
		return genai.FinishReasonSafety
	case "function_call":
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonStop
	}
}

func applyGenerationConfig(params *openai.ChatCompletionNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.TopK != nil {
		return fmt.Errorf("top_k is not supported")
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: cfg.StopSequences}
	}
	if cfg.CandidateCount > 1 {
		params.N = param.NewOpt(int64(cfg.CandidateCount))
	}
	if cfg.FrequencyPenalty != nil {
		params.FrequencyPenalty = param.NewOpt(float64(*cfg.FrequencyPenalty))
	}
	if cfg.PresencePenalty != nil {
		params.PresencePenalty = param.NewOpt(float64(*cfg.PresencePenalty))
	}
	if cfg.ResponseLogprobs {
		if cfg.Logprobs != nil {
			params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		} else {
			params.TopLogprobs = param.NewOpt(int64(1))
		}
	}
	if cfg.SystemInstruction != nil {
		inst := covertSystemMessage(cfg.SystemInstruction)
		params.Messages = append(params.Messages, inst...)
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" {
		return fmt.Errorf("response_mime_type is not supported")
	}
	return nil
}
