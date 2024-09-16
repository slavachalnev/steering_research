import React, { useState, useRef, useEffect } from "react";
import { TokenDisplay } from "./FeatureCard";
import { getBaseUrl } from "./utils";

interface TestSamplesProps {
	feature: number;
	maxAct: number;
}

interface TestSampleProps {
	id: string;
	feature: number;
	maxAct: number;
	removeSample: (id: string) => void;
}

export const TestSample: React.FC<TestSampleProps> = ({
	id,
	feature,
	maxAct,
	removeSample,
}) => {
	const [testText, setTestText] = useState("These are ancient sumerian texts");
	const [showTest, setShowTest] = useState<boolean>(false);
	const [loading, setLoading] = useState<boolean>(false);
	const [hovering, setHovering] = useState<boolean>(false);
	const testTextRef = useRef<HTMLDivElement>(null);
	const [testActivations, setTestActivations] = useState<any>([
		[
			[0],
			[0],
			[2.229448080062866],
			[0.4582373797893524],
			[0.826278805732727],
			[1.7092781066894531],
			[1.4163553714752197],
		],
	]);
	const [textTokens, setTextTokens] = useState<string[]>([
		"These",
		"▁are",
		"▁ancient",
		"▁sum",
		"er",
		"ian",
		"▁texts",
	]);

	const handleInput = (e: React.FormEvent<HTMLDivElement>) => {
		const newText = e.currentTarget.textContent || "";
		if (newText !== testText) {
			setTestText(newText);
		}
	};

	const submitTest = async (e: React.FormEvent<HTMLDivElement>) => {
		setLoading(true);

		try {
			const url = `${getBaseUrl()}/get_max_feature_acts?text=${encodeURIComponent(
				testText
			)}&features=${feature}`;
			const response = await fetch(url, {
				method: "GET",
				headers: {
					"Content-Type": "application/json",
				},
			});

			if (!response.ok) {
				throw new Error("Network response was not ok");
			}

			const data = await response.json();
			console.log("Max feature acts data:", data);
			setTestActivations(data.activations);
			setTextTokens(data.tokens);
		} catch (error) {
			console.error("Error fetching max feature acts:", error);
		} finally {
			setLoading(false);
			setShowTest(false);
		}
	};

	useEffect(() => {
		if (testTextRef.current) {
			testTextRef.current.textContent = testText;
		}
	}, []); // This effect runs only once on mount

	return (
		<div
			style={{
				position: "relative",
				marginTop: "6px",
				minHeight: "24px",
				height: "auto",
				fontSize: ".75rem",
				color: "black",

				backgroundColor: "rgba(240, 240, 240, 1)",
				padding: "4px",
				borderRadius: "8px",
			}}
		>
			<div
				style={{
					height: "auto",
				}}
				onClick={() => {
					setShowTest(true);
					setTimeout(() => {
						testTextRef.current?.focus();
					}, 100);
				}}
				onMouseEnter={() => {
					setHovering(true);
				}}
				onMouseLeave={() => {
					setHovering(false);
				}}
			>
				<div
					ref={testTextRef}
					contentEditable={true}
					style={{
						width: "100%",
						maxWidth: "100%",
						minWidth: "100%",
						borderRadius: "4px",
						padding: "3px",
						textAlign: "left",
						lineHeight: "1.5",
						borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
						// fontSize: "1rem",
						cursor: "text",
						border: "0px solid transparent",
						whiteSpace: "pre-wrap",
						minHeight: "18px",
						outline: "none",
						overflowWrap: "break-word",
						display: showTest ? "block" : "none",
					}}
					onFocus={() => setShowTest(true)}
					onBlur={() => setShowTest(false)}
					onInput={handleInput}
					onKeyDown={(e) => {
						if (e.key === "Enter" && !e.shiftKey) {
							e.preventDefault();
							submitTest(e);
						}
					}}
				/>
				{testActivations && textTokens && !showTest && (
					<div
						style={{
							width: "100%",
							maxWidth: "100%",
							minWidth: "100%",
							borderRadius: "4px",
							padding: "3px",
							textAlign: "left",
							lineHeight: "1.5",
							color: "black",
							height: "auto",
							overflowWrap: "break-word",
							position: "static",
							userSelect: "none",
							top: 0,
							left: 0,
							right: 0,
							bottom: 0,
							zIndex: 1,
						}}
					>
						<div style={{ display: "inline-block" }}>
							{textTokens.map((token: string, index: number) => (
								<TokenDisplay
									key={index}
									index={index}
									token={token}
									value={testActivations[0][index.toString()][0]}
									maxValue={maxAct}
								/>
							))}
						</div>
					</div>
				)}
				{hovering && (
					<div
						style={{
							position: "absolute",
							bottom: "3px",
							right: "3px",
							cursor: "pointer",
							color: "rgba(0, 0, 0, 0.5)",
						}}
						onClick={() => removeSample(id)}
					>
						<svg
							width="16"
							height="16"
							viewBox="0 0 16 16"
							fill="none"
							xmlns="http://www.w3.org/2000/svg"
						>
							<path
								d="M3 8H13"
								stroke="currentColor"
								strokeWidth="1.5"
								strokeLinecap="round"
								strokeLinejoin="round"
							/>
						</svg>
					</div>
				)}
				{showTest && (
					<div
						style={{
							position: "absolute",
							bottom: "3px",
							right: "3px",
							cursor: "pointer",
							color: "rgba(0, 0, 0, 0.5)",
						}}
						onClick={() => testTextRef.current?.blur()}
					>
						{loading ? (
							<svg
								width="16"
								height="16"
								viewBox="0 0 16 16"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
							>
								<path
									d="M8 1.5V4.5M8 11.5V14.5M3.5 8H0.5M15.5 8H12.5M13.3 13.3L11.1 11.1M13.3 2.7L11.1 4.9M2.7 13.3L4.9 11.1M2.7 2.7L4.9 4.9"
									stroke="currentColor"
									strokeWidth="1.5"
									strokeLinecap="round"
									strokeLinejoin="round"
								>
									<animateTransform
										attributeName="transform"
										type="rotate"
										from="0 8 8"
										to="360 8 8"
										dur="1s"
										repeatCount="indefinite"
									/>
								</path>
							</svg>
						) : (
							<svg
								width="16"
								height="16"
								viewBox="0 0 16 16"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
							>
								<path
									d="M3 8H13M13 8L8 3M13 8L8 13"
									stroke="currentColor"
									strokeWidth="1.5"
									strokeLinecap="round"
									strokeLinejoin="round"
								/>
							</svg>
						)}
					</div>
				)}
			</div>
		</div>
	);
};

export const TestSamples: React.FC<TestSamplesProps> = ({
	feature,
	maxAct,
}) => {
	const [samples, setSamples] = useState<string[]>([crypto.randomUUID()]);

	const removeSample = (id: string) => {
		if (samples.length === 1 && samples[0] === id) {
			setSamples([crypto.randomUUID()]);
		} else {
			setSamples(samples.filter((sample) => sample !== id));
		}
	};

	return (
		<div>
			{samples.map((sample) => {
				return (
					<TestSample
						key={sample}
						id={sample}
						feature={feature}
						maxAct={maxAct}
						removeSample={removeSample}
					/>
				);
			})}
		</div>
	);
};

export default TestSamples;
