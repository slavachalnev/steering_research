import React, { useState, useEffect } from "react";
import TokenDisplay from "./TokenDisplay";
import FeatureColumn from "./FeatureColumn";
import { getBaseUrl } from "../utils";

import bret_max_activations from "../data/base_bret_max_activations.json";

interface InspectorProps {
	tokens: string[];
	activations: number[][];
	processedFeatures: ProcessedFeaturesType[];
	containerWidth: string;
	padding: string;
}

interface ProcessedFeaturesType {
	feature: number;
	id: string;
	feature_results: FeatureData[];
}

interface FeatureData {
	binMax: number;
	binMin: number;
	maxValue: number;
	minValue: number;
	tokens: string[];
	values: number[];
}

const Inspector: React.FC<InspectorProps> = ({
	tokens,
	activations,
	processedFeatures,
	containerWidth,
	padding,
}) => {
	const [magnified, setMagnified] = useState<number>(-1);
	const [maxActivations, setMaxActivations] = useState<Record<string, number>>(
		{}
	);
	const [focusToken, setFocusToken] = useState<number>(-1);
	const [processedTokenFeatures, setProcessedTokenFeatures] = useState<
		ProcessedFeaturesType[]
	>([]);

	useEffect(() => {
		const maxActivations = processedFeatures.map((feature) => ({
			[feature.feature]: feature.feature_results.reduce((max, current) =>
				current.maxValue > max.maxValue ? current : max
			).binMax,
		}));
		const result = Object.assign({}, ...maxActivations);
		setMaxActivations(result);
	}, [processedFeatures]);

	const onMagnify = (id: string) => {
		const index: number = processedFeatures.findIndex(
			(feature) => feature.id === id
		);
		setMagnified(index !== -1 ? processedFeatures[index].feature : -1);
	};

	const removeFeature = (id: string) => {
		// This function might need to be lifted to the parent component
		// if it needs to modify the processedFeatures state in App.tsx
	};

	const inspectToken = async (index: number) => {
		if (focusToken === index) {
			setFocusToken(-1);
			setProcessedTokenFeatures([]);
		} else {
			setFocusToken(index);
			setMagnified(-1);
			// You might need to adjust this part based on how you're getting bret_max_activations
			let tokenFeatures = bret_max_activations
				.slice(1)
				[index].top_activations.map((activation) => activation.index);

			const data = await get_activations(tokenFeatures);
			setProcessedTokenFeatures(data);
		}
	};

	const get_activations = async (features: number[]) => {
		const url = `${getBaseUrl()}/get_feature?features=${features.join(",")}`;
		const response = await fetch(url, {
			method: "GET",
			headers: {
				"Content-Type": "application/json",
			},
		});
		const data = await response.json();
		return data.map((d: any) => ({
			...d,
			id: crypto.randomUUID(),
		}));
	};

	return (
		<>
			<div
				style={{
					position: "relative",
					padding: `${padding} 0`,
					textAlign: "left",
					fontSize: ".75rem",
					borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
					userSelect: "none",
					overflow: "hidden",
					color: "white",
					width: containerWidth,
				}}
			>
				<div style={{ display: "inline-block" }}>
					{tokens.slice(1).map((token: string, i: number) => {
						const magnifiedIndex = processedFeatures.findIndex(
							(feature) => feature.feature === magnified
						);
						const value =
							magnified !== -1 ? activations[i + 1][magnifiedIndex] : 0;
						const maxValue = magnified !== -1 ? maxActivations[magnified] : 0;
						return (
							<TokenDisplay
								key={i}
								index={i}
								token={token}
								value={focusToken !== -1 ? (focusToken === i ? 1 : 0) : value}
								maxValue={focusToken !== -1 ? 1 : maxValue}
								backgroundColor={focusToken !== -1 ? "211, 56,43" : undefined}
								color={"white"}
								fontSize={"18px"}
								inspectToken={inspectToken}
							/>
						);
					})}
				</div>
			</div>
			{focusToken === -1 && processedFeatures.length > 0 && (
				<FeatureColumn
					onMagnify={onMagnify}
					processedFeatures={processedFeatures}
					removeFeature={removeFeature}
				/>
			)}
			{processedTokenFeatures.length > 0 && (
				<FeatureColumn
					onMagnify={onMagnify}
					processedFeatures={processedTokenFeatures}
					removeFeature={removeFeature}
				/>
			)}
		</>
	);
};

export default Inspector;
