import React, { useState, useEffect } from "react";
import TokenDisplay from "./TokenDisplay";

import {
	CONTAINER_WIDTH,
	BORDER_RADIUS,
	FONT_SIZE,
	PADDING,
	processText,
	getActivations,
	getMaxActivatingFeatures,
} from "../utils";
import { ProcessedFeaturesType } from "../types";
import bret_tokens from "../data/base_bret_tokens_small.json";
import bret_max_activations from "../data/base_bret_max_activations.json";
//
const Inspector = ({
	processedFeatures,
	setProcessedFeatures,
	uniqueFeatures,
	setUniqueFeatures,
	magnified,
	setMagnified,
}: {
	processedFeatures: ProcessedFeaturesType[];
	setProcessedFeatures: React.Dispatch<
		React.SetStateAction<ProcessedFeaturesType[]>
	>;
	uniqueFeatures: ProcessedFeaturesType[];
	setUniqueFeatures: React.Dispatch<
		React.SetStateAction<ProcessedFeaturesType[]>
	>;
	magnified: number | null;
	setMagnified: React.Dispatch<React.SetStateAction<number | null>>;
}) => {
	const [text, setText] = useState("");
	const [processingState, setProcessingState] = useState("");

	// Number of tokens in text
	const [tokens, setTokens] = useState(bret_tokens.tokens);

	// Activations for top tokens on text
	// TODO: This should be a dictionary mapping, indexing into this leads to bugs
	const [activations, setActivations] = useState(bret_tokens.activations);

	const [maxTokenActivations, setMaxTokenActivations] =
		useState<any>(bret_max_activations);
	const [maxActivations, setMaxActivations] = useState<Record<string, number>>(
		{}
	);

	const [focusToken, setFocusToken] = useState<number | null>(null);

	const updateFields = (data: any) => {
		setActivations(data.activations);
		setTokens(data.tokens);
	};

	const resetFields = () => {
		setActivations([]);
		setTokens([]);
	};

	const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (text.trim() != "") {
			resetFields();

			setProcessingState("Getting unique activations for text");
			const data = await processText(text);
			console.log(data);

			if (data.error) {
				setProcessingState("Error processing text");
				setTimeout(() => {
					setProcessingState("");
				}, 2000);
				return;
			}

			setProcessingState("Data received");
			updateFields(data);

			setProcessingState("Getting feature visualizations");
			const features = data.features;
			const activations = await getActivations(features);
			console.log(activations);
			setProcessingState("");

			setProcessedFeatures(activations);
			setUniqueFeatures(activations);
			setText("");

			const maxTokenActivations = await getMaxActivatingFeatures(
				data.tokens.slice(1).join("")
			);
			console.log(maxTokenActivations);
			setMaxTokenActivations(maxTokenActivations);
		}
	};

	const inspectToken = async (index: number) => {
		if (focusToken == index) {
			setFocusToken(null);
			console.log(uniqueFeatures);
			setProcessedFeatures(uniqueFeatures);
		} else {
			setFocusToken(index);
			setMagnified(null);

			// console.log(tokens.join(""));
			// console.log(activations);
			// console.log(bret_max_activations);

			// These are the top activations for the token
			let tokenFeatures = maxTokenActivations
				.slice(1)
				[index].top_activations.map((activation: any) => activation.index);

			console.log(tokenFeatures);

			setProcessingState("Getting feature visualizations");
			const data = await getActivations(tokenFeatures);

			console.log(data);
			setProcessingState("");
			setProcessedFeatures(data);
		}
	};

	useEffect(() => {
		const maxActivations = processedFeatures.map((feature) => {
			return {
				[feature.feature]: feature.feature_results.reduce((max, current) =>
					current.maxValue > max.maxValue ? current : max
				).binMax,
			};
		});
		const result = Object.assign({}, ...maxActivations);

		console.log("MAX ACTIVATIONS");
		console.log(result);
		setMaxActivations(result);
	}, [processedFeatures]);

	return (
		<div style={{ width: CONTAINER_WIDTH }}>
			<form
				onSubmit={handleSubmit}
				style={{ marginTop: "60px", width: "100%" }}
			>
				<textarea
					value={text}
					onChange={(e) => setText(e.target.value)}
					placeholder="Paste your passage here"
					rows={5}
					style={{
						width: "100%",
						marginBottom: "10px",
						borderRadius: BORDER_RADIUS,
						fontSize: FONT_SIZE,
						padding: PADDING,
					}}
				/>
				<div
					style={{
						display: "flex",
						flexDirection: "row",
					}}
				>
					<button
						type="submit"
						style={{
							backgroundColor: "white",
							width: "fit-content",
							border: "none",
							color: "grey",
							textAlign: "center",
							textDecoration: "none",
							display: "inline-block",
							margin: "4px 2px",
							cursor: "pointer",
							whiteSpace: "nowrap",
							borderRadius: BORDER_RADIUS,
							fontSize: FONT_SIZE,
							padding: PADDING,
						}}
					>
						Process Passage
					</button>
					<div style={{ width: CONTAINER_WIDTH }}>{processingState}</div>
				</div>
			</form>
			<div
				style={{
					position: "relative",
					padding: `${PADDING} 0`,
					textAlign: "left",
					fontSize: ".75rem",
					borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
					userSelect: "none",
					overflow: "hidden",
					color: "white",
					width: CONTAINER_WIDTH,
				}}
			>
				<div style={{ display: "inline-block" }}>
					{tokens &&
						tokens.slice(1).map((token: string, i: number) => {
							const magnifiedIndex = processedFeatures.findIndex(
								(feature) => feature.feature === magnified
							);
							const value = magnified ? activations[i + 1][magnifiedIndex] : 0;
							const maxValue = magnified ? maxActivations[magnified] : 0;
							return (
								<TokenDisplay
									key={i}
									index={i}
									token={token}
									value={focusToken ? (focusToken == i ? 1 : 0) : value}
									maxValue={focusToken ? 1 : maxValue}
									backgroundColor={focusToken ? "211, 56,43" : undefined}
									color={"white"}
									fontSize={"18px"}
									inspectToken={inspectToken}
								/>
							);
						})}
				</div>
			</div>
		</div>
	);
};

export default Inspector;
