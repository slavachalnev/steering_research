import { useState, useEffect, useRef } from "react";
import "./App.css";
// import { keyframes } from "@emotion/react";

const getBaseUrl = () => {
	return process.env.NODE_ENV === "development"
		? "http://localhost:5000"
		: "https://steering-explorer-server.vercel.app";
};

const TokenDisplay = ({ token, value }: { token: string; value: number }) => {
	const opacity = Math.min(0.6, value / 60);
	const [isHovering, setIsHovering] = useState(false);
	const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
	const spanRef = useRef<HTMLSpanElement>(null);

	// Remove "▁" if it's the first character, otherwise keep the token as is
	const displayToken = token.startsWith("▁") ? token.slice(1) : token;
	const addSpace = token.includes("▁") ? " " : "";

	const updateTooltipPosition = () => {
		if (spanRef.current) {
			const rect = spanRef.current.getBoundingClientRect();
			setTooltipPosition({
				top: rect.top - 24, // 24px above the span
				left: rect.left + rect.width / 2,
			});
		}
	};

	return (
		<span
			ref={spanRef}
			style={{
				position: "relative",
				paddingLeft: addSpace ? "4px" : "0px",
				display: "inline-block",
			}}
			onMouseEnter={() => {
				setIsHovering(true);
				updateTooltipPosition();
			}}
			onMouseLeave={() => setIsHovering(false)}
		>
			{isHovering && (
				<div
					style={{
						position: "fixed",
						top: `${tooltipPosition.top}px`,
						left: `${tooltipPosition.left}px`,
						transform: "translateX(-50%)",
						backgroundColor: "rgba(0, 0, 0, 0.8)",
						color: "white",
						borderRadius: "4px",
						fontSize: "12px",
						whiteSpace: "nowrap",
						padding: "2px 4px",
						zIndex: 1000,
						pointerEvents: "none",
					}}
				>
					{value.toFixed(2)}
				</div>
			)}
			<span
				style={{
					backgroundColor: `rgba(0, 0, 255, ${opacity})`,
					display: "inline-block",
					borderRadius: "4px",
					color: "black",
					// color: opacity > 0.6 ? "white" : "black",
				}}
			>
				{addSpace}
				{displayToken}
			</span>
		</span>
	);
};

const ActivationItem = ({ activation }: { activation: any }) => {
	// Find the index of the token with the highest value
	const maxValueIndex = activation.values.indexOf(
		Math.max(...activation.values)
	);

	// Calculate the start index, ensuring it's not negative
	const startIndex = Math.max(0, maxValueIndex - 10);

	return (
		<div
			style={{
				position: "relative",
				paddingBottom: "7px",
				paddingTop: "7px",
				textAlign: "left",
				fontSize: ".75rem",
				borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
				userSelect: "none",
				overflow: "hidden",
				whiteSpace: "nowrap",
			}}
		>
			<div style={{ display: "inline-block" }}>
				{activation.tokens.slice(startIndex).map((token: string, i: number) => (
					<TokenDisplay
						key={i + startIndex}
						token={token}
						value={activation.values[i + startIndex]}
					/>
				))}
			</div>
		</div>
	);
};

function FeatureCard({
	id,
	featureNumber,
	onDelete,
}: {
	id: string;
	featureNumber: number;
	onDelete: (id: string) => void;
}) {
	const [activations, setActivations] = useState([]);
	const [description, setDescription] = useState("");
	const [expanded, setExpanded] = useState(false);
	const [contentHeight, setContentHeight] = useState("auto");
	const contentRef = useRef<HTMLDivElement>(null);
	const [opacity, setOpacity] = useState(0);

	const [loading, setLoading] = useState(true);

	const fetchFeatureData = async () => {
		try {
			const response = await fetch(
				`${getBaseUrl()}/api/feature/${featureNumber}`,
				{
					method: "GET",
					headers: {
						"Content-Type": "application/json",
					},
				}
			);
			const data = await response.json();
			// Filter out duplicate activations
			const uniqueActivations = data.activations.filter(
				(activation: any, index: number, self: any) =>
					index ===
					self.findIndex(
						(t: any) => t.tokens.join("") === activation.tokens.join("")
					)
			);
			setLoading(false);
			setActivations(uniqueActivations);
			setDescription(data.explanations[0].description);
		} catch (error) {
			console.error("Error fetching feature data:", error);
		}
	};

	useEffect(() => {
		fetchFeatureData();
	}, [featureNumber]);

	useEffect(() => {
		if (contentRef.current) {
			setContentHeight(
				expanded ? `${contentRef.current.scrollHeight}px` : "0px"
			);
		}
	}, [expanded, activations]);

	useEffect(() => {
		if (expanded) {
			setOpacity(0);
			setTimeout(() => setOpacity(1), 50);
		} else {
			setOpacity(0);
		}
	}, [expanded]);

	return (
		<div
			style={{
				backgroundColor: "rgba(255,255,255, 0.8)",
				padding: "5px",
				borderRadius: "4px",
				minWidth: "400px",
				maxWidth: "650px",
				position: "relative",
			}}
		>
			<div
				style={{
					position: "absolute",
					top: "5px",
					right: "5px",
					cursor: "pointer",
					fontSize: "16px",
					color: "gray",
					transition: "color 0.1s ease-in-out",
				}}
				onClick={() => onDelete(id)}
				onMouseEnter={(e) => (e.currentTarget.style.color = "black")}
				onMouseLeave={(e) => (e.currentTarget.style.color = "gray")}
			>
				<svg
					width="16"
					height="16"
					viewBox="0 0 16 16"
					fill="none"
					xmlns="http://www.w3.org/2000/svg"
				>
					<path
						d="M2 4h12M5.333 4V2.667a1.333 1.333 0 011.334-1.334h2.666a1.333 1.333 0 011.334 1.334V4m2 0v9.333a1.333 1.333 0 01-1.334 1.334H4.667a1.333 1.333 0 01-1.334-1.334V4h9.334z"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
				</svg>
			</div>
			<div
				style={{
					display: "flex",
					flexDirection: "row",
					color: "black",
				}}
			>
				<div
					style={{
						borderRadius: "5px",
						backgroundColor: "rgba(0, 0, 255, 0.65)",
						padding: "1px",
						fontSize: ".75rem",
						width: "fit-content",
						height: "fit-content",
						whiteSpace: "nowrap",
						color: "white",
					}}
				>
					Feature {featureNumber}
				</div>
				<div
					style={{
						fontSize: ".75rem",
						marginLeft: "5px",
						marginRight: "10px",
						padding: "1px",
						fontWeight: "bold",
						textAlign: "left",
						color: "rgba(0, 0, 0, 0.5)",
					}}
				>
					{description}
					{loading && <span className="loading-text">Loading</span>}
					{activations.length == 0 && !loading && "No activations found"}
				</div>
			</div>

			<div>
				{activations.slice(0, 3).map((activation: any, index: number) => (
					<ActivationItem key={index} activation={activation} />
				))}
			</div>

			<div
				ref={contentRef}
				style={{
					transition: "height 0.3s ease-in-out, opacity 0.5s ease-in-out",
					height: contentHeight,
					overflow: "hidden",
					opacity: opacity,
				}}
			>
				{activations.slice(3, 10).map((activation: any, index: number) => (
					<ActivationItem key={index + 3} activation={activation} />
				))}
			</div>

			{activations.length > 3 && (
				<div
					onClick={() => setExpanded(!expanded)}
					style={{
						// marginTop: "10px",
						// padding: "5px",
						cursor: "pointer",
						background: "none",
						border: "none",
						fontSize: "16px",
						lineHeight: 1,
						transition: "transform 0.3s ease",
						transform: expanded ? "rotate(-180deg)" : "rotate(0deg)",
						color: "black",
						userSelect: "none",
					}}
					aria-label={expanded ? "Collapse" : "Expand"}
				>
					▼
				</div>
			)}
		</div>
	);
}

export default FeatureCard;
