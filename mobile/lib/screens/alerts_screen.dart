import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:hive_flutter/hive_flutter.dart';
import '../services/socket_service.dart';

class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key});

  @override
  State<AlertsScreen> createState() => _AlertsScreenState();
}

class _AlertsScreenState extends State<AlertsScreen> {
  final _socketService = SocketService();
  List<Map<String, dynamic>> _alerts = [];
  Box? _alertsBox;
  StreamSubscription? _subscription;

  // Throttling variables
  final List<Map<String, dynamic>> _bufferedAlerts = [];
  Timer? _throttleTimer;
  DateTime _lastSnackBarTime = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _initializeAlerts();
  }

  Future<void> _initializeAlerts() async {
    _alertsBox = await Hive.openBox('alerts');

    if (_alertsBox != null) {
      final storedAlerts = _alertsBox!.values.toList();
      if (mounted) {
        setState(() {
          _alerts = storedAlerts
              .map((e) => Map<String, dynamic>.from(e))
              .toList();

          _alerts.sort((a, b) {
            DateTime timeA =
                DateTime.tryParse(a['timestamp'] ?? '') ?? DateTime.now();
            DateTime timeB =
                DateTime.tryParse(b['timestamp'] ?? '') ?? DateTime.now();
            return timeB.compareTo(timeA);
          });

          if (_alerts.length > 50) {
            _alerts = _alerts.sublist(0, 50);
          }
        });
      }
    }

    _connectSocket();
  }

  Future<void> _clearAlerts() async {
    await _alertsBox?.clear();
    if (mounted) {
      setState(() {
        _alerts.clear();
      });
    }
  }

  void _connectSocket() {
    _socketService.connect();
    _subscription = _socketService.stream.listen(
      (data) {
        _handleIncomingData(data);
      },
      onError: (error) {
        print('Socket error: $error');
      },
    );
  }

  void _handleIncomingData(dynamic data) {
    try {
      final parsedData = jsonDecode(data.toString());
      final List<dynamic> messages = parsedData['messages'] ?? [];
      final String timestamp =
          parsedData['timestamp'] ?? DateTime.now().toString();
      final String cleanMessage = messages.isNotEmpty
          ? messages.join('\n')
          : 'Unknown Alert';

      final newAlert = {
        'message': cleanMessage,
        'timestamp': timestamp,
        'raw': data.toString(),
      };

      _bufferedAlerts.add(newAlert);

      if (_throttleTimer == null || !_throttleTimer!.isActive) {
        _throttleTimer = Timer(
          const Duration(milliseconds: 500),
          _processBufferedAlerts,
        );
      }
    } catch (e) {
      print('Error parsing alert data: $e');
    }
  }

  void _processBufferedAlerts() {
    if (!mounted || _bufferedAlerts.isEmpty) return;

    final List<Map<String, dynamic>> alertsToProcess = List.from(
      _bufferedAlerts,
    );
    _bufferedAlerts.clear();

    setState(() {
      for (final alert in alertsToProcess) {
        final msg = (alert['message'] ?? '').toString().toLowerCase();
        if (msg.contains('low soil moisture')) {
          _alerts.removeWhere(
            (a) => (a['message'] ?? '').toString().toLowerCase().contains(
              'low soil moisture',
            ),
          );
          _alertsBox?.put('latest_soil_moisture', alert);
        } else {
          _alertsBox?.add(alert);
        }
        _alerts.insert(0, alert);
      }

      // Keep list size manageable
      if (_alerts.length > 50) {
        _alerts = _alerts.sublist(0, 50);
      }
    });

    // Show SnackBar only for the latest alert and throttle it
    if (alertsToProcess.isNotEmpty) {
      final latestAlert = alertsToProcess.last;
      final now = DateTime.now();
      if (now.difference(_lastSnackBarTime).inSeconds >= 2) {
        _lastSnackBarTime = now;
        ScaffoldMessenger.of(context).clearSnackBars();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              latestAlert['message'],
              style: const TextStyle(color: Colors.white),
            ),
            backgroundColor: Colors.red.shade800,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(10),
            ),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    }
  }

  @override
  void dispose() {
    _throttleTimer?.cancel();
    _subscription?.cancel();
    _socketService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D0D), // Dark background
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'Alerts',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        iconTheme: const IconThemeData(color: Colors.white),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline),
            onPressed: _clearAlerts,
            tooltip: 'Clear History',
          ),
        ],
      ),
      body: _alerts.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.notifications_off_outlined,
                    size: 64,
                    color: Colors.white.withOpacity(0.5),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'No alerts yet',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.5),
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _alerts.length,
              itemBuilder: (context, index) {
                final alert = _alerts[index];
                // Try to parse timestamp for better formatting
                String timeDisplay = alert['timestamp'] ?? '';
                try {
                  final dt = DateTime.parse(timeDisplay);
                  timeDisplay =
                      "${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}";
                } catch (e) {
                  // Keep original string if parse fails
                }

                return Container(
                  margin: const EdgeInsets.only(bottom: 12),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1E1E1E), // Darker card background
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: Colors.red.withOpacity(0.5),
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.2),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: ListTile(
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 8,
                    ),
                    leading: Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: Colors.red.withOpacity(0.1),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.warning_amber_rounded,
                        color: Colors.redAccent,
                        size: 28,
                      ),
                    ),
                    title: Text(
                      alert['message'] ?? 'Unknown Alert',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                        fontSize: 16,
                      ),
                    ),
                    subtitle: Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Row(
                        children: [
                          Icon(
                            Icons.access_time,
                            size: 14,
                            color: Colors.white.withOpacity(0.4),
                          ),
                          const SizedBox(width: 4),
                          Text(
                            timeDisplay,
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.4),
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
    );
  }
}
