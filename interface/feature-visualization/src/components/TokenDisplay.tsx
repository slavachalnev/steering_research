import { useMemo, useState, useRef, useEffect } from "react";

const TokenDisplay = ({
	index,
	token,
	value,
	maxValue,
	color = "black",
	backgroundColor = "42, 97, 211",
	fontSize = ".75rem",
	inspectToken = (id: number) => {},
}: {
	index: number;
	token: string;
	value: number;
	maxValue: number;
	color?: string;
	backgroundColor?: string;
	fontSize?: string;
	inspectToken?: (id: number) => void;
}) => {
	const [isHovering, setIsHovering] = useState(false);
	const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
	const spanRef = useRef<HTMLSpanElement>(null);

	// Use useMemo to recalculate opacity when value or maxValue changes
	const opacity = useMemo(() => {
		if (value == 0 && maxValue == 0) return 0;
		return Math.min(0.85, value / maxValue);
	}, [value, maxValue]);

	const displayToken = useMemo(
		() =>
			token.startsWith("▁") || token.startsWith(" ") ? token.slice(1) : token,
		[token]
	);

	const addSpace = useMemo(
		() => (token.includes("▁") || token.startsWith(" ") ? " " : ""),
		[token]
	);

	const updateTooltipPosition = () => {
		if (spanRef.current) {
			const rect = spanRef.current.getBoundingClientRect();
			setTooltipPosition({
				top: rect.top - 24, // 24px above the span
				left: rect.left + rect.width / 2,
			});
		}
	};

	// Update tooltip position when value changes
	useEffect(() => {
		if (isHovering) {
			updateTooltipPosition();
		}
	}, [value, isHovering]);

	return (
		<span
			ref={spanRef}
			style={{
				position: "relative",
				paddingLeft: addSpace ? `${3.25}px` : "0px",
				display: "inline-block",
			}}
			onMouseEnter={() => {
				setIsHovering(true);
				updateTooltipPosition();
			}}
			onMouseLeave={() => setIsHovering(false)}
			onClick={() => {
				inspectToken(index);
			}}
		>
			{" "}
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
						// fontSize: "12px",
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
					backgroundColor: `rgba(${backgroundColor}, ${opacity})`,
					display: "inline-block",
					borderRadius: "4px",
					fontSize,
					color,
				}}
			>
				{addSpace}
				{displayToken}
			</span>
		</span>
	);
};

export default TokenDisplay;
