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

class Braider {
private:
    // Nothing
public:
    // Making config parameters global
    int nailCount;
    int stringCount;
    float stringWidth;
    float stringIntensity;
    int deadZone;
    int width;
    int height;
    float radius;
    float invRadiusSquared;
    vector<vector<float>> grayData;
    vector<sf::Vector2f> nailPositions;
    vector<size_t> nailIndices;

    Braider(
        int _nailCount,
        int _stringCount,
        float _stringWidth,
        float _stringIntensity,
        int _deadZone,
        vector<vector<float>> _grayData
    ) {
        nailCount = _nailCount;
        stringCount = _stringCount;
        stringWidth = _stringWidth;
        stringIntensity = _stringIntensity;
        deadZone = _deadZone;

        // Initialize grayData
        grayData = _grayData;

        // Get sizes
        width = grayData.size();
        height = grayData[0].size();

        // Make image into a circle
        radius = width / 2;
        invRadiusSquared = 1.0f / (radius * radius);
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

        // Invert image
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                grayData[i][j] = 1.0f - grayData[i][j];
            }
        }

        // Get nail positions
        getNailPositions(radius);

        // Initialize nail indices
        nailIndices = vector<size_t>(stringCount + 1, 0);
    }

    void getNailPositions(float radius) {
        // Calculate the angle between nails
        float angle = 2 * M_PI / nailCount;

        // Calculate the positions of the nails
        nailPositions.resize(nailCount);
        for (int i = 0; i < nailCount; i++) {
            float x = radius + radius * cos(i * angle);
            float y = radius + radius * sin(i * angle);
            nailPositions[i] = sf::Vector2f(x, y);
        }
    }

    float tryString(
        sf::Vector2f s, sf::Vector2f e,
        float stringWidth, float stringIntensity,
        bool apply = false
    ) {
        // Get bounding box
        int min_x = max(int(floor(min(s.x, e.x) - stringWidth)), 0);
        int max_x = min(int(ceil(max(s.x, e.x) + stringWidth)), width - 1);
        int min_y = max(int(floor(min(s.y, e.y) - stringWidth)), 0);
        int max_y = min(int(ceil(max(s.y, e.y) + stringWidth)), height - 1);
        
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
                    float newValue = grayData[x][y] - smoothstep(stringWidth, 0.0f, distance) * stringIntensity;
                    loss += (newValue*newValue - grayData[x][y]*grayData[x][y]);
                    if (apply) {
                        grayData[x][y] = newValue;
                    }
                }
            }
        }
        loss /= sqrt((e.x - s.x) * (e.x - s.x) + (e.y - s.y) * (e.y - s.y));

        return loss;
    }

    float greedyStep(int i) {
        // Get the previous nail
        size_t prevNail = nailIndices[i - 1];

        // Find the nail that minimizes the loss
        float minLoss = 1e9;
        size_t minNail = 0;
        for(int j = 0; j < nailCount; j++) {
            int dist = min({abs((int)j - (int)prevNail), abs((int)j - (int)prevNail - (int)nailCount), abs((int)j - (int)prevNail + (int)nailCount)});
            if(j == prevNail || dist <= deadZone || (i > 1 && j == nailIndices[i - 2])) {
                continue;
            }
            float loss = tryString(nailPositions[prevNail], nailPositions[j], stringWidth, stringIntensity);
            if(loss < minLoss) {
                minLoss = loss;
                minNail = j;
            }
        }

        tryString(nailPositions[prevNail], nailPositions[minNail], stringWidth, stringIntensity, true);

        nailIndices[i] = minNail;

        return minLoss;
    }

    void greedy() {
        // Get nail sequence using greedy approach
        for(int i = 1; i <= stringCount; i++) {
            greedyStep(i);
        }
    }

    float computeTotalLoss() {
        float totalLoss = 0.0f;
        for (int i = 0; i < grayData.size(); i++) {
            for (int j = 0; j < grayData[0].size(); j++) {
                totalLoss += grayData[i][j] * grayData[i][j];
            }
        }
        return totalLoss;
    }

    void process() {
        // Use gready approach
        greedy();
        cout << "Greedy total loss: " << computeTotalLoss() << endl;
    }
};

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    void getNailIndices(void* braider_ptr, int* output) {
        // Convert void pointer to Braider object
        Braider* braider = (Braider*)braider_ptr;

        // Convert nail indices to array
        for (int i = 0; i < braider->stringCount + 1; i++) {
            output[i] = braider->nailIndices[i];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void doGreedyStep(void* braider_ptr, int i) {
        // Convert void pointer to Braider object
        Braider* braider = (Braider*)braider_ptr;

        // Do greedy step
        braider->greedyStep(i);
        return;
    }

    EMSCRIPTEN_KEEPALIVE
    void doGreedy(void* braider_ptr) {
        // Convert void pointer to Braider object
        Braider* braider = (Braider*)braider_ptr;

        // Do greedy
        braider->greedy();
        return;
    }

    EMSCRIPTEN_KEEPALIVE
    void* createBraider(
        int _nailCount,
        int _stringCount,
        float _stringWidth,
        float _stringIntensity,
        float _gamma,
        float _brightness,
        float _edgeDarkening,
        int _deadZone,
        float* data,
        int height,
        int width
    ) {
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
                grayData[i][j] = pow(grayData[i][j], _gamma) * _brightness;
            }
        }

        // Apply edge detection
        auto edgeData = sobelEdgeDetection(grayData);

        // Darken the edges
        for (int i = 0; i < grayData.size(); i++) {
            for (int j = 0; j < grayData[0].size(); j++) {
                grayData[i][j] *= 1.0f - edgeData[i][j] * _edgeDarkening;
            }
        }

        // Transposing the vector to make "x" coordinate the first index
        grayData = transpose(grayData);

        Braider* braider = new Braider(
            _nailCount,
            _stringCount,
            _stringWidth,
            _stringIntensity,
            _deadZone,
            grayData
        );

        return (void*)braider;
    }

    EMSCRIPTEN_KEEPALIVE
    void deleteBraider(void* braider) {
        delete (Braider*)braider;
    }
}