import React, { useState, useRef, useEffect } from "react";
import TokenDisplay from "./TokenDisplay";
import { getBaseUrl } from "../utils";
import { FeatureCardSubHeader } from "./FeatureCard";
import { LoadingIcon, MinusIcon, PlusIcon, RightArrowIcon } from "./Icons";

interface TestSamplesProps {
	feature: number;
	maxAct: number;
}

interface TestSampleProps {
	id: string;
	feature: number;
	maxAct: number;
	removeSample: (id: string) => void;
	onlySample: boolean;
}

export const TestSample: React.FC<TestSampleProps> = ({
	id,
	feature,
	maxAct,
	removeSample,
	onlySample,
}) => {
	// const [testText, setTestText] = useState("These are ancient sumerian texts");
	const [testText, setTestText] = useState("");
	const [showInput, setShowInput] = useState<boolean>(false);
	const [loading, setLoading] = useState<boolean>(false);
	const [hovering, setHovering] = useState<boolean>(false);
	const testTextRef = useRef<HTMLDivElement>(null);
	// const [testActivations, setTestActivations] = useState<any>([
	// 	[
	// 		[0],
	// 		[0],
	// 		[2.229448080062866],
	// 		[0.4582373797893524],
	// 		[0.826278805732727],
	// 		[1.7092781066894531],
	// 		[1.4163553714752197],
	// 	],
	// ]);
	// const [textTokens, setTextTokens] = useState<string[]>([
	// 	"These",
	// 	"▁are",
	// 	"▁ancient",
	// 	"▁sum",
	// 	"er",
	// 	"ian",
	// 	"▁texts",
	// ]);
	const [testActivations, setTestActivations] = useState<any>([]);
	const [textTokens, setTextTokens] = useState<string[]>([]);

	const handleInput = (e: React.FormEvent<HTMLDivElement>) => {
		const newText = e.currentTarget.textContent || "";
		if (newText !== testText) {
			setTestText(newText);
		}
	};

	const submitTest = async () => {
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
			testTextRef.current?.blur();
			setShowInput(false);
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
					if (!showInput) {
						setShowInput(true);
						setTimeout(() => {
							testTextRef.current?.focus();
						}, 100);
					}
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
					contentEditable={loading ? false : true}
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
						display: showInput ? "block" : "none",
					}}
					onFocus={() => setShowInput(true)}
					// onBlur={() => {
					// 	setTimeout(() => {
					// 		if (!loading) {
					// 			setShowInput(false);
					// 		}
					// 	}, 300);
					// }}
					onInput={handleInput}
					onKeyDown={(e) => {
						if (e.key === "Enter" && !e.shiftKey) {
							e.preventDefault();
							submitTest();
						}
					}}
				></div>
				{testText == "" && !showInput && (
					<div
						style={{
							position: "absolute",
							top: 7,
							left: 7,
							right: 0,
							bottom: 0,
							zIndex: 1,
							color: "grey",
						}}
					>
						{"Enter text to test sample"}
					</div>
				)}
				{testActivations && textTokens && !showInput && (
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
				{hovering &&
					((onlySample && testText !== "") || !onlySample) &&
					!showInput && (
						<MinusIcon
							onClick={() => {
								removeSample(id);
							}}
							style={{
								position: "absolute",
								bottom: "8px",
								right: "4px",
								zIndex: 1,
							}}
						/>
					)}
				{showInput && (
					<div
						style={{
							position: "absolute",
							bottom: "3px",
							right: "3px",
							cursor: "pointer",
							color: "rgba(0, 0, 0, 0.5)",
						}}
					>
						{loading ? (
							<LoadingIcon />
						) : (
							<RightArrowIcon
								onClick={() => {
									console.log("submitTest");
									submitTest();
								}}
							/>
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
			<div
				style={{
					display: "flex",
					flexDirection: "row",
				}}
			>
				<FeatureCardSubHeader text={"Test samples"} />
				<PlusIcon
					onClick={() => setSamples([...samples, crypto.randomUUID()])}
					style={{
						cursor: "pointer",
						marginLeft: "8px",
						marginTop: "8px",
					}}
				/>
			</div>
			{samples.map((sample, i) => {
				return (
					<TestSample
						key={sample}
						id={sample}
						feature={feature}
						maxAct={maxAct}
						removeSample={removeSample}
						onlySample={samples.length === 1}
					/>
				);
			})}
		</div>
	);
};

export default TestSamples;
