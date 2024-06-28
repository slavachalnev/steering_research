import React, { useState } from "react";
import { steeringData } from "../utils/api";
import { MixturePreview } from "./FeatureMixer";

const Result = ({ result }) => {
	const [selectedResult, setSelectedResult] = useState(0);
	const [hoveredResult, setHoveredResult] = useState(null);
	const prompt = result.metadata.prompt;

	const displayResult = hoveredResult !== null ? hoveredResult : selectedResult;

	return (
		<div className="result">
			<div className="result-header">
				<div className="result-steering">
					{/* {JSON.parse(result.metadata.steering).map((direction) => {
						return (
							<div className="result-steering-direction">
								{steeringData[direction[0]]
									? steeringData[direction[0]]
									: direction[0]}{" "}
								- {direction[1]}
							</div>
						);
					})} */}
					<MixturePreview
						features={result.features}
						style={{
							width: "200px",
							height: "18px",
							marginBottom: "0px",
						}}
					/>
				</div>
				<div className="result-variations">
					{result.texts.map((_, i) => {
						const variationClasses = `result-variation ${
							i === selectedResult ? "result-selected" : ""
						} ${i === hoveredResult ? "result-hovered" : ""}`;

						return (
							<div
								className={variationClasses}
								onMouseEnter={() => {
									if (i != selectedResult) setHoveredResult(i);
								}}
								onMouseLeave={() => setHoveredResult(null)}
								onClick={() => setSelectedResult(i)}
							>
								{i}
							</div>
						);
					})}
				</div>
			</div>

			<div className="result-texts">
				{result.texts.map((text, i) => {
					const classes = `result-text result-text-${i}`;
					const isSelected = i === displayResult;

					return isSelected ? (
						<div className={classes}>
							<span
								style={{
									fontWeight: "bold",
									color: "black",
								}}
							>
								{prompt}
							</span>
							<span
								dangerouslySetInnerHTML={{
									__html: text.slice(prompt.length),
								}}
							/>
						</div>
					) : null;
				})}
			</div>
		</div>
	);
};

const Results = ({ results }) => {
	return (
		<div>
			{[...results].reverse().map((result) => {
				return <Result result={result} />;
			})}
		</div>
	);
};

export default Results;
