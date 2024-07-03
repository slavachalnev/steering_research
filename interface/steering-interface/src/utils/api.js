export const steeringData = {
	10138: "London",
	2378: "Wedding",
	11067: "Poetry",
	10200: "San Francisco",
	6831: "Political Scandals",
	3169: "Gratitude",
};

const baseURL = "http://192.168.5.99:5000";
export const predictData = async (data) => {
	const url = baseURL + "/predict";

	try {
		// Sending the initial request to the /predict endpoint
		let response = await fetch(url, {
			method: "POST",
			headers: {
				// Authorization: `Bearer ${api_key}`,
				"Content-Type": "application/json",
			},
			body: JSON.stringify(data),
		});

		if (!response.ok) {
			throw new Error(`Initial request failed: ${response.statusText}`);
		}

		let responseData = await response.json();

		// Extracting the prediction ID for polling
		let predictionId = responseData.id;
		let status = responseData.status;

		// Polling the API until the prediction is complete
		while (status !== "succeeded" && status !== "failed") {
			await new Promise((resolve) => setTimeout(resolve, 5000)); // Wait for 5 seconds before polling again

			let pollUrl = `${url}/${predictionId}`;
			response = await fetch(pollUrl, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
			});

			if (!response.ok) {
				throw new Error(`Polling request failed: ${response.statusText}`);
			}

			responseData = await response.json();
			status = responseData.status;
			console.log(`Prediction status: ${status}`);
		}

		// Return the final response
		if (status === "succeeded") {
			return {
				status: "succeeded",
				output: responseData.output,
			};
		} else {
			throw new Error(`Prediction failed: ${responseData.error}`);
		}
	} catch (error) {
		console.error("Error:", error);
		return {
			status: "failed",
			error: error.message,
		};
	}
};

export const getNeuronpedia = async (feature) => {
	const url = baseURL + "/get-neuron/" + feature;

	try {
		const res = await fetch(url, {
			headers: {
				accept: "*/*",
			},
			method: "GET",
		});
		if (!res.ok) {
			throw new Error(`Request failed with status ${res.status}`);
		}
		const json = await res.json();
		console.log(json);
		return json;
	} catch (error) {
		console.error("Error fetching data:", error);
		return { error: error.message };
	}
};
