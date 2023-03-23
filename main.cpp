#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "emscripten.h"

#define EPS 1e-6

using namespace std;

namespace sf {
    struct Vector2f {
        float x, y;

        Vector2f() : x(0), y(0) {}
        Vector2f(float x, float y) : x(x), y(y) {}

        Vector2f operator+(const Vector2f& other) const {
            return Vector2f(x + other.x, y + other.y);
        }

        Vector2f operator-(const Vector2f& other) const {
            return Vector2f(x - other.x, y - other.y);
        }

        Vector2f operator*(float scalar) const {
            return Vector2f(x * scalar, y * scalar);
        }

        Vector2f operator/(float scalar) const {
            return Vector2f(x / scalar, y / scalar);
        }
    };
}

// Making config parameters global
int nailCount;
int stringCount;
float stringWidth;
float stringIntensity;
float gammaValue;
float brightness;
float edgeDarkening;
int seed;
int genAlgoIterations;
int maxNailOffset;

vector<vector<float>> transpose(const vector<vector<float>>& matrix) {
    int height = matrix.size();
    int width = matrix[0].size();

    vector<vector<float>> transposed(width, vector<float>(height));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

vector<sf::Vector2f> getNailPositions(float radius) {
    // Calculate the angle between nails
    float angle = 2 * M_PI / nailCount;

    // Calculate the positions of the nails
    vector<sf::Vector2f> nailPositions(nailCount);
    for (int i = 0; i < nailCount; i++) {
        float x = radius + radius * cos(i * angle);
        float y = radius + radius * sin(i * angle);
        nailPositions[i] = sf::Vector2f(x, y);
    }

    return nailPositions;
}

float clamp(float x, float lower, float upper) {
    return max(lower, min(x, upper));
}

float smoothstep(float edge0, float edge1, float x) {
    // Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    // Evaluate polynomial
    return x * x * (3 - 2 * x);
}

float dot(sf::Vector2f a, sf::Vector2f b) {
    return a.x * b.x + a.y * b.y;
}

float lineSDF(sf::Vector2f point, sf::Vector2f a, sf::Vector2f b) {
    // Computes the signed distance from a point to a line segment
    sf::Vector2f ab = b - a;
    sf::Vector2f ap = point - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0f, 1.0f);
    sf::Vector2f closestPoint = a + ab * t;
    return sqrt(dot(point - closestPoint, point - closestPoint));
}

float tryString(
    const vector<vector<float>>& grayData,
    vector<vector<float>>& newGrayData,
    sf::Vector2f s, sf::Vector2f e,
    float stringWidth, float stringIntensity,
    bool apply = false
) {
    // Get bounding box
    int min_x = max(int(floor(min(s.x, e.x) - stringWidth)), 0);
    int max_x = min(int(ceil(max(s.x, e.x) + stringWidth)), (int)grayData.size() - 1);
    int min_y = max(int(floor(min(s.y, e.y) - stringWidth)), 0);
    int max_y = min(int(ceil(max(s.y, e.y) + stringWidth)), (int)grayData[0].size() - 1);

    // Get radius
    float radius = grayData.size() / 2.0f;
    
    // Compute the range in the "y" direction that string can cover
    float angle = atan2(abs(e.y - s.y), abs(e.x - s.x));
    float cosAngle = cos(angle);
    int offset;
    if (abs(cosAngle) < EPS) {
        offset = max_y - min_y;
    } else {
        offset = ceil(stringWidth / cosAngle);
    }

    // Check the values of the pixels in the bounding box, but optimizing using offset
    float loss = 0.0f;
    for(int x = min_x; x <= max_x; x++) {
        int sy;
        if(e.x - s.x == 0) {
            sy = (max_y + min_y) / 2;
        } else {
            sy = s.y + (x - s.x) * (e.y - s.y) / (e.x - s.x);
        }

        for(int y = max(sy - offset, min_y); y <= min(sy + offset, max_y); y++) {
            float distance = lineSDF(sf::Vector2f(x + 0.5f, y + 0.5f), s, e);
            if (distance < stringWidth) {
                float newValue = newGrayData[x][y] - smoothstep(stringWidth, 0.0f, distance) * stringIntensity;
                //newValue = max(newValue, 0.0f);
                float closs = newValue - grayData[x][y];
                float ploss = newGrayData[x][y] - grayData[x][y];
                float cdis = (x - radius) * (x - radius) + (y - radius) * (y - radius);
                loss += (closs*closs - ploss*ploss) * (1.5f - cdis / radius / radius);
                if (apply) {
                    newGrayData[x][y] = newValue;
                }
            }
        }
    }
    loss /= sqrt((e.x - s.x) * (e.x - s.x) + (e.y - s.y) * (e.y - s.y));

    return loss;
}

vector<size_t> greedy(
    const vector<vector<float>>& grayData,
    vector<vector<float>>& newGrayData,
    const vector<sf::Vector2f>& nailPositions
) {
    // Get nail sequence
    vector<vector<uint8_t>> used(nailCount, vector<uint8_t>(nailCount, 0)); // Repetition of strings is not allowed
    vector<size_t> nailSequence(stringCount + 1, 0);
    size_t prevNail = 0;
    for(int i = 1; i <= stringCount; i++) {
        // Print progress every 10 steps
        // if (i % 10 == 0) {
        //     cout << "\rGreedy: " << i << "/" << stringCount << " strings" << flush;
        // }
        
        float minLoss = 1e9;
        size_t minNail = 0;
        for(int j = 0; j < nailCount; j++) {
            if(j == prevNail || used[prevNail][j]) {
                continue;
            }
            float loss = tryString(grayData, newGrayData, nailPositions[prevNail], nailPositions[j], stringWidth, stringIntensity);
            if(loss < minLoss) {
                minLoss = loss;
                minNail = j;
            }
        }

        tryString(grayData, newGrayData, nailPositions[prevNail], nailPositions[minNail], stringWidth, stringIntensity, true);

        // Mark pair of nails as used
        //used[prevNail][minNail] = 1;
        //used[minNail][prevNail] = 1;

        nailSequence[i] = minNail;
        prevNail = minNail;
    }
    // cout << endl;

    return nailSequence;
}

void genAlgo(
    const vector<vector<float>>& grayData,
    vector<vector<float>>& newGrayData,
    const vector<sf::Vector2f>& nailPositions,
    vector<size_t>& nailIndices
) {
    // Set random seed
    srand(seed);

    for(int iter_id = 0; iter_id < genAlgoIterations; iter_id++) {
        // if(iter_id % 100 == 0) {
        //     cout << "\rGenAlgo: " << iter_id << "/" << genAlgoIterations << " iterations" << flush;
        // }

        // Select random nail, except first and last
        size_t nailIndex = rand() % (nailCount - 2) + 1;

        // Generate new nail id
        int offset = rand() % (2 * maxNailOffset) - maxNailOffset;
        if (offset >= 0) {
            offset++;
        }
        size_t newNail = (nailIndices[nailIndex] + offset) % nailCount;

        // Compute how much loss will be removed by removing two strings attached to this nail
        float removeLoss = 0.0f;
        removeLoss += tryString(grayData, newGrayData, nailPositions[nailIndices[nailIndex - 1]], nailPositions[nailIndices[nailIndex]], stringWidth, -stringIntensity, true);
        removeLoss += tryString(grayData, newGrayData, nailPositions[nailIndices[nailIndex]], nailPositions[nailIndices[nailIndex + 1]], stringWidth, -stringIntensity, true);

        // Compute how much loss will be added by adding two strings attached to new nail
        float addLoss = 0.0f;
        addLoss += tryString(grayData, newGrayData, nailPositions[nailIndices[nailIndex - 1]], nailPositions[newNail], stringWidth, stringIntensity, true);
        addLoss += tryString(grayData, newGrayData, nailPositions[newNail], nailPositions[nailIndices[nailIndex + 1]], stringWidth, stringIntensity, true);

        if (addLoss + removeLoss < 0.0f) {
            // Accept new nail
            nailIndices[nailIndex] = newNail;
        } else {
            // Reject new nail
            tryString(grayData, newGrayData, nailPositions[nailIndices[nailIndex - 1]], nailPositions[newNail], stringWidth, -stringIntensity, true);
            tryString(grayData, newGrayData, nailPositions[newNail], nailPositions[nailIndices[nailIndex + 1]], stringWidth, -stringIntensity, true);
            tryString(grayData, newGrayData, nailPositions[nailIndices[nailIndex - 1]], nailPositions[nailIndices[nailIndex]], stringWidth, stringIntensity, true);
            tryString(grayData, newGrayData, nailPositions[nailIndices[nailIndex]], nailPositions[nailIndices[nailIndex + 1]], stringWidth, stringIntensity, true);
        }
    }
    // cout << endl;

    return;
}

float computeTotalLoss(const vector<vector<float>>& grayData, const vector<vector<float>>& newGrayData) {
    float totalLoss = 0.0f;
    for (int i = 0; i < grayData.size(); i++) {
        for (int j = 0; j < grayData[0].size(); j++) {
            totalLoss += (grayData[i][j] - newGrayData[i][j]) * (grayData[i][j] - newGrayData[i][j]);
        }
    }
    return totalLoss;
}

vector<size_t> process(vector<vector<float>> grayData) {
    // Get sizes
    int width = grayData.size();
    int height = grayData[0].size();

    // Make image into a circle
    float radius = width / 2;
    float radiusSquared = radius * radius;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            float x = i - radius;
            float y = j - radius;
            float distanceSquared = x * x + y * y;
            if (distanceSquared > radiusSquared) {
                grayData[i][j] = 1.0f;
            }
        }
    }

    // Create new image that will be modified by strings
    vector<vector<float>> newGrayData(width, vector<float>(height, 1.0f));

    // Compute nail positions
    vector<sf::Vector2f> nailPositions = getNailPositions(radius);

    // Use gready approach
    vector<size_t> nailIndices = greedy(grayData, newGrayData, nailPositions);
    cout << "Greedy total loss: " << computeTotalLoss(grayData, newGrayData) << endl;

    // Use genetic algorithm
    genAlgo(grayData, newGrayData, nailPositions, nailIndices);
    cout << "Genetic algorithm total loss: " << computeTotalLoss(grayData, newGrayData) << endl;

    return nailIndices;
}

vector<vector<float>> sobelEdgeDetection(const vector<vector<float>>& image) {
    int height = image.size();
    int width = image[0].size();

    // Create two 3x3 kernels for horizontal and vertical edges
    vector<vector<float>> horizontalKernel = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    vector<vector<float>> verticalKernel = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    // Create a new image to store the edge detection result
    vector<vector<float>> result(height, vector<float>(width));

    // Apply the Sobel operator to each pixel in the image
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float horizontalGradient = 0.0f;
            float verticalGradient = 0.0f;

            // Convolve the kernels with the image
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    horizontalGradient += image[i + k][j + l] * horizontalKernel[k + 1][l + 1];
                    verticalGradient += image[i + k][j + l] * verticalKernel[k + 1][l + 1];
                }
            }

            // Calculate the magnitude of the gradient
            float gradientMagnitude = sqrt(horizontalGradient * horizontalGradient + verticalGradient * verticalGradient);

            // Set the pixel value in the result image
            result[i][j] = clamp(gradientMagnitude, 0.0f, 1.0f);
        }
    }

    return result;
}

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    int getNailIndices(float* data, int height, int width, int* output) {
        // Convert data to vector
        vector<vector<float>> grayData(height, vector<float>(width));
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                grayData[i][j] = data[i * width + j];
            }
        }
        
        // Apply gamma and brightness
        for (int i = 0; i < grayData.size(); i++) {
            for (int j = 0; j < grayData[0].size(); j++) {
                grayData[i][j] = pow(grayData[i][j], gammaValue) * brightness;
            }
        }

        if (edgeDarkening > 0.0f) {
            // Apply edge detection
            auto edgeData = sobelEdgeDetection(grayData);

            // Darken the edges
            for (int i = 0; i < grayData.size(); i++) {
                for (int j = 0; j < grayData[0].size(); j++) {
                    grayData[i][j] *= 1.0f - edgeData[i][j] * edgeDarkening;
                }
            }
        }

        // Transposing the vector to make "x" coordinate the first index
        grayData = transpose(grayData);

        // Process image
        vector<size_t> nailIndices = process(grayData);

        // Convert nail indices to array
        for (int i = 0; i < stringCount + 1; i++) {
            output[i] = nailIndices[i];
        }

        return EXIT_SUCCESS;
    }

    EMSCRIPTEN_KEEPALIVE
    void setConfigParameters(
        int _nailCount,
        int _stringCount,
        float _stringWidth,
        float _stringIntensity,
        float _gamma,
        float _brightness,
        float _edgeDarkening,
        int _seed,
        int _genAlgoIterations,
        int _maxNailOffset
    ) {
        nailCount = _nailCount;
        stringCount = _stringCount;
        stringWidth = _stringWidth;
        stringIntensity = _stringIntensity;
        gammaValue = _gamma;
        brightness = _brightness;
        edgeDarkening = _edgeDarkening;
        seed = _seed;
        genAlgoIterations = _genAlgoIterations;
        maxNailOffset = _maxNailOffset;
    }

    EMSCRIPTEN_KEEPALIVE
    int hello() {
        return nailCount;
    }
}