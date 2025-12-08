import 'package:flutter/material.dart';

class YourFieldsScreen extends StatefulWidget {
  const YourFieldsScreen({super.key});

  @override
  State<YourFieldsScreen> createState() => _YourFieldsScreenState();
}

class _YourFieldsScreenState extends State<YourFieldsScreen> {
  final List<String> _fields = ['Field 1', 'Field 2', 'Field 3'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Your Fields',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: ListView.builder(
                itemCount: _fields.length,
                itemBuilder: (context, index) {
                  return Container(
                    margin: const EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: const Color(0xFF3E3E3E),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      title: Text(
                        _fields[index],
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                        ),
                      ),
                      trailing: const Icon(
                        Icons.arrow_forward_ios,
                        color: Colors.white,
                        size: 16,
                      ),
                      onTap: () {
                        // Navigate to field details
                      },
                    ),
                  );
                },
              ),
            ),
            Container(
              width: double.infinity,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF3E3E3E),
                borderRadius: BorderRadius.circular(12),
              ),
              child: ListTile(
                title: const Text(
                  'Add Field',
                  style: TextStyle(color: Colors.white, fontSize: 16),
                ),
                trailing: const Icon(Icons.add, color: Colors.white),
                onTap: () {
                  // Add new field logic
                  setState(() {
                    _fields.add('Field ${_fields.length + 1}');
                  });
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
