import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../services/ai_service.dart';

class AiAssistantScreen extends StatefulWidget {
  const AiAssistantScreen({super.key});

  @override
  State<AiAssistantScreen> createState() => _AiAssistantScreenState();
}

class _AiAssistantScreenState extends State<AiAssistantScreen> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final AiService _aiService = AiService();
  final List<Map<String, String>> _messages = [];
  late String _sessionId;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    // _sessionId = DateTime.now().millisecondsSinceEpoch.toString();
    // Add initial greeting
    _messages.add({'role': 'ai', 'content': 'ai_greeting'.tr()});
  }

  Future<void> _sendMessage() async {
    if (_controller.text.trim().isEmpty) return;

    final query = _controller.text.trim();
    setState(() {
      _messages.add({'role': 'user', 'content': query});
      _isLoading = true;
    });
    _controller.clear();
    _scrollToBottom();

    Map<String, dynamic> yieldContext = {};
    try {
      final box = await Hive.openBox('last_recommendation');
      if (box.containsKey('data')) {
        final data = box.get('data');
        if (data is Map) {
          yieldContext = Map<String, dynamic>.from(data);
        }
      }
    } catch (e) {
      print('Error loading yield context: $e');
    }

    try {
      final response = await _aiService.chat(
        sessionId: "token", // Service will replace this with actual token
        query: query,
        language: context.locale.languageCode,
        yieldContext: yieldContext,
      );

      if (response != null) {
        final content =
            response['response'] ??
            response['answer'] ??
            response['message'] ??
            response['reply'] ??
            response['content'];

        if (content != null) {
          setState(() {
            _messages.add({'role': 'ai', 'content': content.toString()});
          });
        } else {
          print('AI Response missing content key: $response');
          setState(() {
            _messages.add({'role': 'ai', 'content': 'no_response'.tr()});
          });
        }
      } else {
        setState(() {
          _messages.add({'role': 'ai', 'content': 'no_response'.tr()});
        });
      }
    } catch (e) {
      setState(() {
        _messages.add({'role': 'ai', 'content': 'connection_error'.tr()});
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D0D),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E1E1E),
        elevation: 0,
        title: Text(
          'ai_assistant'.tr(),
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                final isUser = message['role'] == 'user';
                return Align(
                  alignment: isUser
                      ? Alignment.centerRight
                      : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.only(bottom: 16),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                    constraints: BoxConstraints(
                      maxWidth: MediaQuery.of(context).size.width * 0.75,
                    ),
                    decoration: BoxDecoration(
                      color: isUser
                          ? const Color(0xFFC5E1A5)
                          : const Color(0xFF1E1E1E),
                      borderRadius: BorderRadius.only(
                        topLeft: const Radius.circular(16),
                        topRight: const Radius.circular(16),
                        bottomLeft: isUser
                            ? const Radius.circular(16)
                            : Radius.zero,
                        bottomRight: isUser
                            ? Radius.zero
                            : const Radius.circular(16),
                      ),
                      border: isUser ? null : Border.all(color: Colors.white24),
                    ),
                    child: MarkdownBody(
                      data: message['content'] ?? '',
                      styleSheet: MarkdownStyleSheet(
                        p: TextStyle(
                          color: isUser ? Colors.black : Colors.white,
                          fontSize: 16,
                        ),
                        strong: TextStyle(
                          color: isUser ? Colors.black : Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                        em: TextStyle(
                          color: isUser ? Colors.black : Colors.white,
                          fontStyle: FontStyle.italic,
                        ),
                        h1: TextStyle(
                          color: isUser ? Colors.black : Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                        h2: TextStyle(
                          color: isUser ? Colors.black : Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                        h3: TextStyle(
                          color: isUser ? Colors.black : Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                        code: TextStyle(
                          color: isUser ? Colors.black87 : Colors.white70,
                          backgroundColor: isUser
                              ? Colors.white.withOpacity(0.3)
                              : Colors.black.withOpacity(0.3),
                          fontFamily: 'monospace',
                        ),
                        codeblockDecoration: BoxDecoration(
                          color: isUser
                              ? Colors.white.withOpacity(0.3)
                              : Colors.black.withOpacity(0.3),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        listBullet: TextStyle(
                          color: isUser ? Colors.black : Colors.white,
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(color: Color(0xFFC5E1A5)),
            ),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: const BoxDecoration(
              color: Color(0xFF1E1E1E),
              border: Border(top: BorderSide(color: Colors.white10)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    style: const TextStyle(color: Colors.white),
                    decoration: InputDecoration(
                      hintText: 'ask_anything'.tr(),
                      hintStyle: TextStyle(
                        color: Colors.white.withOpacity(0.5),
                      ),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(24),
                        borderSide: BorderSide.none,
                      ),
                      filled: true,
                      fillColor: Colors.white.withOpacity(0.1),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 10,
                      ),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                const SizedBox(width: 8),
                CircleAvatar(
                  backgroundColor: const Color(0xFFC5E1A5),
                  child: IconButton(
                    icon: const Icon(Icons.send, color: Colors.black),
                    onPressed: _sendMessage,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
