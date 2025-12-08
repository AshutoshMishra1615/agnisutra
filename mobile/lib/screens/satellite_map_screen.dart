import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:geolocator/geolocator.dart';
import '../services/satellite_service.dart';

class SatelliteMapScreen extends StatefulWidget {
  const SatelliteMapScreen({super.key});

  @override
  State<SatelliteMapScreen> createState() => _SatelliteMapScreenState();
}

class _SatelliteMapScreenState extends State<SatelliteMapScreen> {
  final MapController _mapController = MapController();
  LatLng? _currentCenter;
  bool _isLoadingLocation = true;
  String? _errorMessage;

  double? ndviValue;
  String? overlayImageUrl;
  // Bounds for the overlay (Your backend must provide these corners)
  LatLngBounds? imageBounds;

  @override
  void initState() {
    super.initState();
    _initializeLocation();
  }

  Future<void> _initializeLocation() async {
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          if (mounted) {
            setState(() {
              _isLoadingLocation = false;
              _errorMessage = "Location permission denied";
            });
          }
          return;
        }
      }

      if (permission == LocationPermission.deniedForever) {
        if (mounted) {
          setState(() {
            _isLoadingLocation = false;
            _errorMessage =
                "Location permission denied forever. Please enable it in settings.";
          });
        }
        return;
      }

      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      if (mounted) {
        setState(() {
          _currentCenter = LatLng(position.latitude, position.longitude);
          _isLoadingLocation = false;
        });
        _mapController.move(_currentCenter!, 13.0);
        _loadData(_currentCenter!);
      }
    } catch (e) {
      print("Error getting location: $e");
      if (mounted) {
        setState(() {
          _isLoadingLocation = false;
          _errorMessage = "Error getting location: $e";
        });
      }
    }
  }

  void _loadData(LatLng center) async {
    final service = SatelliteService();
    final data = await service.getSatelliteData(
      center.latitude,
      center.longitude,
    );

    if (mounted) {
      setState(() {
        ndviValue = data['ndvi_value'];
        overlayImageUrl = data['image_url'];

        // Parse bounds from backend response [[lat1, lon1], [lat2, lon2]]
        if (data['bounds'] != null) {
          var b = data['bounds'];
          imageBounds = LatLngBounds(
            LatLng(b[0][0], b[0][1]),
            LatLng(b[1][0], b[1][1]),
          );
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoadingLocation) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    if (_errorMessage != null || _currentCenter == null) {
      return Scaffold(
        appBar: AppBar(title: const Text("Crop Health (NDVI)")),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.location_off, size: 64, color: Colors.grey),
                const SizedBox(height: 16),
                Text(
                  _errorMessage ?? "Location not available",
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _isLoadingLocation = true;
                      _errorMessage = null;
                    });
                    _initializeLocation();
                  },
                  child: const Text("Retry"),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Crop Health (NDVI)")),
      body: Stack(
        children: [
          // 1. THE MAP
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: _currentCenter!,
              initialZoom: 13.0,
            ),
            children: [
              // Base Layer (OpenStreetMap)
              TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'com.agnisutra.app',
              ),

              // SATELLITE OVERLAY (The "Greenness" Map)
              if (overlayImageUrl != null && imageBounds != null)
                OverlayImageLayer(
                  overlayImages: [
                    OverlayImage(
                      bounds: imageBounds!,
                      imageProvider: NetworkImage(overlayImageUrl!),
                      opacity:
                          0.7, // Semi-transparent so you see roads underneath
                    ),
                  ],
                ),
            ],
          ),

          // 2. THE DATA CARD (Floating at bottom)
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Card(
              color: Colors.white.withOpacity(0.9),
              elevation: 5,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      "Vegetation Index (NDVI)",
                      style: TextStyle(color: Colors.grey),
                    ),
                    const SizedBox(height: 5),

                    // The Big Number
                    Text(
                      ndviValue != null ? ndviValue.toString() : "Loading...",
                      style: TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                        color: (ndviValue ?? 0) > 0.5
                            ? Colors.green
                            : Colors.red,
                      ),
                    ),

                    Text(
                      (ndviValue ?? 0) > 0.5
                          ? "Crop is Healthy 🌱"
                          : "Crop Stress Detected ⚠️",
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
