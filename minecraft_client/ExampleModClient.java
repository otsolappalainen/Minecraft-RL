package com.example;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.entity.effect.StatusEffectInstance;
import net.minecraft.entity.effect.StatusEffects;
import net.minecraft.entity.mob.Monster;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.util.math.BlockPos;
import net.minecraft.block.Block;
import org.java_websocket.server.WebSocketServer;
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import com.google.gson.Gson;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.block.BlockState;
//import net.minecraft.network.packet.s2c.play.InventoryS2CPacket;


import java.net.InetSocketAddress;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import net.minecraft.world.World;
import net.minecraft.util.math.Direction;

/**
 * ExampleModClient is a Minecraft mod that interfaces with a Python training script via WebSockets.
 * It receives action commands, executes them within the game, and sends back the player's state.
 */
public class ExampleModClient implements ClientModInitializer {
    // Change from static final to just final to allow different ports per instance
    private final int PORT;
    
    public ExampleModClient() {
        // Get port from system property, default to 8080-8083 based on instance ID
        int instanceId = Integer.parseInt(System.getProperty("instance.id", "0"));
        this.PORT = 8080 + instanceId;
        
        
        
    }

    private WebSocketServer server;
    private MinecraftClient client;
    private Gson gson;

    private double initialX = Double.NaN;
    private double initialY = Double.NaN;
    private double initialZ = Double.NaN;
    private float initialYaw;
    private float initialPitch;
    private final Random random = new Random();

    // Action execution durations in milliseconds
    private static final int ACTION_DURATION_SHORT = 45;   // 80 milliseconds
    private static final int ACTION_DURATION_MEDIUM = 45;  // 80 milliseconds
    private static final int EXTENDED_ACTION_TIME = 45;    // 80 milliseconds

    // Thread pool executor for managing action execution threads
    private ExecutorService actionExecutor = Executors.newSingleThreadExecutor();

    // List to store broken blocks
    private final List<Map<String, Object>> brokenBlocks = new ArrayList<>();

    // Timing constants
    private static final int STATE_SEND_DELAY = 60;       // Delay before sending state

    // Use AtomicBoolean for thread-safe toggles
    private final AtomicBoolean isSneaking = new AtomicBoolean(false);

    // Add connection state tracking
    private volatile boolean isConnected = false;

    // ScheduledExecutorService for task scheduling
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

    // Map to track key release tasks
    private final Map<KeyBinding, ScheduledFuture<?>> keyReleaseTasks = new ConcurrentHashMap<>();

    // For state sending debouncing
    private ScheduledFuture<?> stateSendTask = null;

    // Debug mode flag
    private static final boolean DEBUG_MODE = true;

    //private static final int OBSERVATION_RADIUS = 4; // Radius around the player

    // Add this at the class level
    //private ScheduledFuture<?> attackTask = null;

    private static final int OBSERVATION_RADIUS_X = 6; // Radius on X-axis
    private static final int OBSERVATION_RADIUS_Z = 6; // Radius on Z-axis
    private static final int OBSERVATION_RADIUS_Y_DOWN = 1; // 1 below the player
    private static final int OBSERVATION_RADIUS_Y_UP = 2;   // 2 above the player

    // Change from static final to just static volatile
    private static volatile int MIN_Y_LEVEL = -63;
    private static volatile int MAX_Y_LEVEL = -62;

    // Add these fields at the class level
    private int actionsSinceLastJump = 0;
    private static final int ACTIONS_REQUIRED_BETWEEN_JUMPS = 20;

    // Add these fields at class level
    private static final boolean INFO_MODE = false;
    private int stateSendCounter = 0;
    private long startTime = System.currentTimeMillis();


    private BlockPos lastAttackedBlock = null;
    private double breakProgress = 0.0;
    private int otherActionsSinceAttack = 0;

    @Override
    public void onInitializeClient() {
        client = MinecraftClient.getInstance();
        gson = new Gson();

        

        // Register block break event listener
        PlayerBlockBreakEvents.AFTER.register((world, player, pos, state, entity) -> {
            if (player.equals(client.player)) {
                Block brokenBlock = state.getBlock();
                String blockName = brokenBlock.getTranslationKey().toLowerCase();
                Map<String, Object> blockInfo = new HashMap<>();

                if (blockName.contains("diamond") || blockName.contains("gold") ||
                    blockName.contains("redstone") || blockName.contains("iron")) {
                    blockInfo.put("blocktype", 1);
                } else if (blockName.contains("stone") && 
                    (pos.getY() >= MIN_Y_LEVEL && pos.getY() <= MAX_Y_LEVEL)) {
                    blockInfo.put("blocktype", 0.6);
                } else {
                    return; // Ignore other blocks
                }

                // Always set coordinates to 0 initially
                blockInfo.put("blockx", 0);
                blockInfo.put("blocky", 0);
                blockInfo.put("blockz", 0);

                // Calculate new blocks revealed
                int newRevealed = calculateNewRevealedBlocks(world, pos);
                double normalizedRevealed = Math.min(newRevealed * 0.2, 1.0);
                blockInfo.put("new_revealed_blocks", normalizedRevealed);

                brokenBlocks.add(blockInfo);
            }
        });

        // Start the WebSocket server
        startWebSocketServer();
    }

    /**
     * Initializes and starts the WebSocket server to listen for action commands.
     */
    private void startWebSocketServer() {
        server = new WebSocketServer(new InetSocketAddress(PORT)) {
            @Override
            public void onOpen(WebSocket conn, ClientHandshake handshake) {
                isConnected = true;
                if (DEBUG_MODE) {
                    System.out.println("[WebSocket] Connection opened on port: " + PORT);
                }
            }

            @Override
            public void onClose(WebSocket conn, int code, String reason, boolean remote) {
                isConnected = false;
                if (DEBUG_MODE) {
                    System.out.println("[WebSocket] Connection closed on port: " + PORT);
                }
                // Do not call cleanup here
            }

            @Override
            public void onMessage(WebSocket conn, String message) {
                if (DEBUG_MODE) {
                    System.out.println("[WebSocket] Received message: " + message + " time=" + System.currentTimeMillis());
                }
                handleAction(conn, message);
            }

            @Override
            public void onError(WebSocket conn, Exception ex) {
                if (DEBUG_MODE) {
                    System.err.println("[WebSocket] Error: " + ex.getMessage());
                    ex.printStackTrace();
                }
            }

            @Override
            public void onStart() {
                if (DEBUG_MODE) {
                    System.out.println("[WebSocket] WebSocket server started successfully on port: " + PORT);
                }
            }
        };

        try {
            server.start();
            if (DEBUG_MODE) {
                System.out.println("[WebSocket] WebSocket server started successfully.");
            }
        } catch (Exception e) {
            if (DEBUG_MODE) {
                System.err.println("[WebSocket] Failed to start WebSocket server: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    /**
     * Handles incoming action messages from the WebSocket client.
     *
     * @param conn    The WebSocket connection.
     * @param message The received JSON message containing the action.
     */
    private synchronized void handleAction(WebSocket conn, String message) {
        long startTime = System.currentTimeMillis();
    
        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> action = gson.fromJson(message, Map.class);
            String actionType = (String) action.get("action");

            ClientPlayerEntity player = client.player;
            if (player != null) {
                // Release attack key before processing new action
                releaseAttackKey();

                // Handle walking conflicts
                if (actionType.startsWith("move_")) {
                    cancelOngoingActionsExcept(Arrays.asList(client.options.sneakKey, client.options.attackKey));
                } else if (!actionType.equals("attack") && !actionType.equals("sneak")) {
                    // Cancel all actions except sneak and attack
                    cancelOngoingActionsExcept(Arrays.asList(client.options.sneakKey, client.options.attackKey));
                }

                if (actionType.startsWith("reset")) {
                    int resetType = Integer.parseInt(actionType.split(" ")[1]);
                    handleReset(player, resetType);
                    // After reset, send state
                    sendPlayerState(conn);
                } else {
                    switch (actionType) {
                        case "move_forward":
                            executeKeyAction(client.options.forwardKey, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "move_backward":
                            executeKeyAction(client.options.backKey, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "move_left":
                            executeKeyAction(client.options.leftKey, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "move_right":
                            executeKeyAction(client.options.rightKey, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "jump_walk_forward":
                            if (actionsSinceLastJump >= ACTIONS_REQUIRED_BETWEEN_JUMPS) {
                                executeJumpWalkForward(client.options.jumpKey, client.options.forwardKey, conn);
                                actionsSinceLastJump = 0; // Reset counter after jump
                            } else {
                                scheduleStateSend(conn);
                            }
                            break;
                        case "jump":
                            if (actionsSinceLastJump >= ACTIONS_REQUIRED_BETWEEN_JUMPS) {
                                executePlayerAction(() -> player.jump(), ACTION_DURATION_MEDIUM, conn);
                                actionsSinceLastJump = 0; // Reset counter after jump
                                otherActionsSinceAttack++;
                                if (otherActionsSinceAttack >= 2) {
                                    breakProgress = 0.0;
                                    lastAttackedBlock = null;
                                }
                            } else {
                                otherActionsSinceAttack++;
                                if (otherActionsSinceAttack >= 2) {
                                    breakProgress = 0.0;
                                    lastAttackedBlock = null;
                                }
                                scheduleStateSend(conn);
                            }
                            break;
                        case "look_left":
                            executeSmoothAdjustYaw(player, -6, ACTION_DURATION_SHORT, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "look_right":
                            executeSmoothAdjustYaw(player, 6, ACTION_DURATION_SHORT, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "look_up":
                            executeSmoothAdjustPitch(player, -6, ACTION_DURATION_SHORT, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "look_down":
                            executeSmoothAdjustPitch(player, 6, ACTION_DURATION_SHORT, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "turn_left":
                            executeSmoothAdjustYaw(player, -10, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "turn_right":
                            executeSmoothAdjustYaw(player, 10, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "next_item":
                            executeItemCycle(player, true, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "previous_item":
                            executeItemCycle(player, false, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "sneak":
                            executeToggleSneak(client.options.sneakKey, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "attack":
                            executeAttackAction(client.options.attackKey, conn);
                            actionsSinceLastJump++;
                            break;
                        case "use":
                            executeKeyAction(client.options.useKey, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "no_op":
                            executeNoOpAction(conn);
                            actionsSinceLastJump++;
                            otherActionsSinceAttack++;
                            if (otherActionsSinceAttack >= 2) {
                                breakProgress = 0.0;
                                lastAttackedBlock = null;
                            }
                            break;
                        case "level_change":
                            try {
                                // Extract min and max values from the action message
                                Double minY = ((Number) action.get("min_y")).doubleValue();
                                Double maxY = ((Number) action.get("max_y")).doubleValue();
                                
                                // Validate the values
                                if (minY != null && maxY != null && minY < maxY) {
                                    MIN_Y_LEVEL = minY.intValue();
                                    MAX_Y_LEVEL = maxY.intValue();
                                    System.out.println("[Y-Levels] Updated: MIN=" + MIN_Y_LEVEL + ", MAX=" + MAX_Y_LEVEL);
                                    
                                }
                            } catch (Exception e) {

                                System.err.println("[Y-Levels] Failed to update Y-levels: " + e.getMessage());
                                
                            }
                            scheduleStateSend(conn);
                            break;
                        // For all other actions, increment the counter
                        default:
                            if (!actionType.startsWith("reset")) { // Don't count resets
                                actionsSinceLastJump++;
                            }
                            if (DEBUG_MODE) {
                                System.out.println("[WebSocket] Unknown action: " + actionType);
                            }
                            // Still send state even if action is unknown
                            sendPlayerState(conn);
                    }
                }
            } else {
                if (DEBUG_MODE) {
                    System.err.println("[WebSocket] Player entity is null. Cannot execute action.");
                }
                // Optionally send an error response
                Map<String, Object> error = new HashMap<>();
                error.put("error", "Player entity is null.");
                conn.send(gson.toJson(error));
            }

            if (DEBUG_MODE) {
                // Add as last line before catch block
                System.out.println("[Timing] Action received at: " + startTime);
            }

        } catch (Exception e) {
            if (DEBUG_MODE) {
                System.err.println("[WebSocket] Error handling action: " + e.getMessage());
                e.printStackTrace();
            }
            // Optionally send an error response
            Map<String, Object> error = new HashMap<>();
            error.put("error", e.getMessage());
            conn.send(gson.toJson(error));
        }
    }

    /**
     * Checks if the Y coordinate is approximately even within a precision of 0.01.
     *
     * @param y The Y coordinate of the player.
     * @return True if Y is approximately even, false otherwise.
     */
    private boolean isYCoordinateEven(double y) {
        double remainder = y % 2.0;
        return Math.abs(remainder) < 0.01 || Math.abs(remainder - 2.0) < 0.01;
    }

    /**
     * Releases the attack key if it is pressed.
     */
    private void releaseAttackKey() {
        KeyBinding attackKey = client.options.attackKey;
        if (attackKey.isPressed()) {
            attackKey.setPressed(false);
            KeyBinding.updatePressedStates();
            if (DEBUG_MODE) {
                System.out.println("[Attack] Attack key released before processing new action.");
            }
        }
    }

    /**
     * Executes a player-specific action (e.g., jump) and schedules state sending.
     *
     * @param action      The action to execute.
     * @param durationMs  Duration in milliseconds to hold the action.
     * @param conn        The WebSocket connection.
     */
    private void executePlayerAction(Runnable action, long durationMs, WebSocket conn) {
        actionExecutor.submit(() -> {
            try {
                action.run();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        scheduleStateSend(conn);
    }

    /**
     * Smoothly adjusts the player's yaw (horizontal rotation) over a specified duration and schedules state sending.
     *
     * @param player      The player entity.
     * @param amount      The total degrees to adjust yaw by.
     * @param durationMs  Total duration in milliseconds over which to adjust yaw.
     * @param conn        The WebSocket connection.
     */
    private void executeSmoothAdjustYaw(ClientPlayerEntity player, float amount, int durationMs, WebSocket conn) {
        float steps = Math.abs(amount);
        float increment = amount / steps;

        actionExecutor.submit(() -> {
            try {
                for (int i = 0; i < steps; i++) {
                    adjustYaw(player, increment);
                    Thread.sleep(durationMs / (int) steps);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        scheduleStateSend(conn);
    }

    /**
     * Smoothly adjusts the player's pitch over a duration, respecting -25/25 degree limits.
     */
    private void executeSmoothAdjustPitch(ClientPlayerEntity player, float amount, int durationMs, WebSocket conn) {
        float steps = Math.abs(amount);
        
        // Calculate adjusted amount if it exceeds limits
        float finalAmount = amount;
        float finalPitch = player.getPitch() + amount;
        if (finalPitch > 80) {
            finalAmount = 80 - player.getPitch();
        } else if (finalPitch < -80) {
            finalAmount = -80 - player.getPitch();
        }
        
        // Use final array to hold the increment value
        final float[] increment = {finalAmount / steps};

        actionExecutor.submit(() -> {
            try {
                for (int i = 0; i < steps; i++) {
                    adjustPitch(player, increment[0]);
                    Thread.sleep(durationMs / (int) steps);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        scheduleStateSend(conn);
    }

    /**
     * Toggles sneak action and schedules state sending.
     *
     * @param key    The key binding for sneaking.
     * @param conn   The WebSocket connection.
     */
    private void executeToggleSneak(KeyBinding key, WebSocket conn) {
        boolean newState = !isSneaking.getAndSet(!isSneaking.get());
        key.setPressed(newState);
        scheduleStateSend(conn);
    }

    /**
     * Executes cycling through inventory items and schedules state sending.
     *
     * @param player  The player entity.
     * @param next    If true, cycles to the next item; otherwise, cycles to the previous item.
     * @param conn    The WebSocket connection.
     */
    private void executeItemCycle(ClientPlayerEntity player, boolean next, WebSocket conn) {
        if (next) {
            player.getInventory().selectedSlot = (player.getInventory().selectedSlot + 1) % 9; // Cycle to next item
        } else {
            player.getInventory().selectedSlot = (player.getInventory().selectedSlot + 8) % 9; // Cycle to previous item
        }
        scheduleStateSend(conn);
    }

    /**
     * Handles reset actions to return the player to a previous state and sends the state after reset.
     *
     * @param player    The player entity.
     * @param resetType The type of reset to perform.
     */
    private void handleReset(ClientPlayerEntity player, int resetType) {
        // Cancel all ongoing actions
        cancelOngoingActionsExcept(null);
    
        // Clear broken blocks
        brokenBlocks.clear();
    
        // Cancel scheduled state send task
        if (stateSendTask != null && !stateSendTask.isDone()) {
            stateSendTask.cancel(false);
            stateSendTask = null;
        }
    
        // Shut down and recreate the action executor
        actionExecutor.shutdownNow();
        try {
            if (!actionExecutor.awaitTermination(100, TimeUnit.MILLISECONDS)) {
                actionExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            actionExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        actionExecutor = Executors.newSingleThreadExecutor();
    
        // Clear key release tasks
        for (ScheduledFuture<?> task : keyReleaseTasks.values()) {
            task.cancel(false);
        }
        keyReleaseTasks.clear();
    
        // Check if player is dead and needs respawn
        if (!player.isAlive()) {
            respawnPlayer();
            try {
                // Give some time for respawn to complete
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    
        switch (resetType) {
            case 0:
                // Stop all actions
                break;
            case 1:
                try {
                    Thread.sleep(50);
                    // Use current position
                    double x = player.getX();
                    double y = player.getY();
                    double z = player.getZ();
                    float yaw = player.getYaw();
                    float pitch = player.getPitch();
                    

                    player.refreshPositionAndAngles(x, y, z, yaw, pitch);
                    player.setVelocity(0, 0, 0);
                    player.setOnGround(true);

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
            }
                break;
            case 2:
                try {
                    Thread.sleep(50);
                    // Use current position
                    double x = player.getX();
                    double y = player.getY();
                    double z = player.getZ();
                    // Randomize yaw between -180 and 180 degrees
                    float yaw = random.nextFloat() * 360 - 180;
                    // Set pitch to 0
                    float pitch = random.nextFloat() * 160 - 80;
    
                    player.refreshPositionAndAngles(x, y, z, yaw, pitch);
                    player.setVelocity(0, 0, 0);
                    player.setOnGround(true);
                    player.setYaw(yaw);
                    player.setPitch(pitch);
    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                break;
            default:
                // Unknown reset type
        }
    
        // Apply Night Vision effect for 5 minutes (6000 ticks)
        applyNightVision(player);
    
        // Reset sneaking state
        isSneaking.set(false);
        client.options.sneakKey.setPressed(false);
    
        // Reset attack key state
        client.options.attackKey.setPressed(false);
        KeyBinding.updatePressedStates();
    
        // Force cleanup of any resources
        System.gc();
    }

    /**
     * Cancels ongoing actions except specified exceptions.
     *
     * @param exceptions List of KeyBindings to exclude from cancellation.
     */
    private void cancelOngoingActionsExcept(List<KeyBinding> exceptions) {
        for (Map.Entry<KeyBinding, ScheduledFuture<?>> entry : keyReleaseTasks.entrySet()) {
            KeyBinding key = entry.getKey();
            if (exceptions != null && exceptions.contains(key)) {
                continue; // Skip exception keys
            }
            ScheduledFuture<?> task = entry.getValue();
            if (task != null && !task.isDone()) {
                task.cancel(false);
            }
            key.setPressed(false);
            KeyBinding.updatePressedStates();
            keyReleaseTasks.remove(key);
        }
    }

    /**
     * Adjusts the player's yaw by a specified amount.
     *
     * @param player The player entity.
     * @param amount The degrees to adjust yaw by.
     */
    private void adjustYaw(ClientPlayerEntity player, float amount) {
        player.setYaw(player.getYaw() + amount);
    }

    /**
     * Adjusts the player's pitch while enforcing limits between -80 and 80 degrees.
     *
     * @param player The player entity.
     * @param amount The degrees to adjust pitch by.
     * @return boolean True if adjustment was made, false if blocked by limits
     */
    private boolean adjustPitch(ClientPlayerEntity player, float amount) {
        float newPitch = player.getPitch() + amount;
        
        // Enforce limits between -80 and 80 degrees
        newPitch = Math.max(-80, Math.min(80, newPitch));
        
        player.setPitch(newPitch);
        return true;
    }

    /**
     * Retrieves the current state of the player.
     *
     * @return A map containing the player's state attributes.
     */
    private Map<String, Object> getPlayerState() {
        Map<String, Object> state = new HashMap<>();
        ClientPlayerEntity player = client.player;

        if (player != null) {
            // Initialize initial position if needed
            if (Double.isNaN(initialX)) {
                initialX = player.getX();
                initialY = player.getY();
                initialZ = player.getZ();
                initialYaw = player.getYaw();
                initialPitch = player.getPitch();
            }

            // Calculate player movement
            double deltaX = player.getX() - initialX;
            double deltaZ = player.getZ() - initialZ;
            initialX = player.getX();
            initialZ = player.getZ();

            // Encode x
            double encodedX = 0.5;
            if (deltaX > 0) {
                encodedX += Math.min(deltaX / 2.0, 0.5);
            } else if (deltaX < 0) {
                encodedX -= Math.min(-deltaX / 2.0, 0.5);
            }

            // Encode z
            double encodedZ = 0.5;
            if (deltaZ > 0) {
                encodedZ += Math.min(deltaZ / 2.0, 0.5);
            } else if (deltaZ < 0) {
                encodedZ -= Math.min(-deltaZ / 2.0, 0.5);
            }

            // Encode y
            double baseY = -64.0; // Y level where encoded value should be 0.5
            double playerY = player.getY();
            double yDiff = playerY - baseY; // Difference from base level
            double scaleFactor = 0.05; // 0.5 change per 10 blocks (0.05 per block)

            // Calculate linear scaling
            double encodedY = 0.5 + (yDiff * scaleFactor);

            // Clip between 0 and 1
            encodedY = Math.max(0.0, Math.min(1.0, encodedY));

            state.put("x", encodedX);
            state.put("y", encodedY);
            
            // Normalize Yaw
            float yaw = ((player.getYaw() % 360 + 360) % 360); // First normalize to 0-360
            if (yaw > 180) {
                yaw -= 360; // Convert to -180 to 180 range
            }
            // Then normalize to 0-1 range for the state
            double normalizedYaw = (yaw + 180) / 360.0;
            state.put("yaw", normalizedYaw);
            
            // Normalize Pitch between -80 and 80 to 0.1 and 0.9
            float pitch = player.getPitch();
            pitch = Math.max(-80, Math.min(80, pitch));
            double normalizedPitch = 0.1 + ((pitch + 80) / 160.0) * 0.8;  // Maps -80 to 0.1 and 80 to 0.9
            state.put("pitch", normalizedPitch);
            
            // Normalize Health between 0 and 20 to 0 and 1
            double health = Math.max(0, Math.min(player.getHealth(), 20));
            double normalizedHealth = health / 20.0;
            state.put("health", normalizedHealth);
            
            // Normalize Light Level between 0 and 15 to 0 and 1
            int lightLevel = client.world.getLightLevel(player.getBlockPos());
            double normalizedLight = Math.max(0, Math.min(lightLevel, 15)) / 15.0;
            state.put("light_level", normalizedLight);
            
            state.put("z", encodedZ);
            
            ItemStack heldItem = player.getMainHandStack();
            double pickaxeValue = 0.0;

            if (!heldItem.isEmpty()) {
                String itemName = heldItem.getName().getString().toLowerCase();
                
                if (itemName.contains("pickaxe")) {
                    if (itemName.contains("wooden")) {
                        pickaxeValue = 0.2;
                    } else if (itemName.contains("stone")) {
                        pickaxeValue = 0.4;
                    } else if (itemName.contains("iron")) {
                        pickaxeValue = 0.6;
                    } else if (itemName.contains("diamond")) {
                        pickaxeValue = 0.8;
                    } else if (itemName.contains("netherite")) {
                        pickaxeValue = 1.0;
                    }
                }
            }

            List<Integer> heldItemArray = Arrays.asList(
                (int)pickaxeValue, // Cast to int since List<Integer> is required
                0, 0, 0,
                0
            );
            state.put("held_item", heldItemArray);

            // Initialize target block array - [block_type, inverse_distance]
            List<Double> targetBlockInfo = Arrays.asList(0.0, 0.0, 0.0);
            state.put("target_block", targetBlockInfo);

            // Get block player is looking at
            double maxReach = 20.0; // Detection range
            double normalReach = 4.4; // Normal interaction reach
            boolean fluid = false;
            var hitResult = player.raycast(maxReach, 0.0f, fluid);

            if (hitResult != null && hitResult.getType() == HitResult.Type.BLOCK) {
                BlockHitResult blockHit = (BlockHitResult) hitResult;
                BlockPos blockPos = blockHit.getBlockPos();
                Block block = client.world.getBlockState(blockPos).getBlock();
                String blockName = block.toString().toLowerCase();

                // Calculate distance
                double dx = blockPos.getX() + 0.5 - player.getX();
                double dy = blockPos.getY() + 0.5 - player.getY();
                double dz = blockPos.getZ() + 0.5 - player.getZ();
                double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

                // Calculate inverse distance (1/d), capped at 1.0 for very close blocks
                double inverseDistance = Math.min(1.0, 1.0 / distance);

                // Conditional check for breakProgress calculation
                if ((blockPos.getY() >= MIN_Y_LEVEL && blockPos.getY() <= MAX_Y_LEVEL) ||
                    blockName.contains("diamond") || blockName.contains("gold") ||
                    blockName.contains("redstone") || blockName.contains("iron")) {

                    // Update break progress based on continuous attacks
                    if (lastAttackedBlock != null && lastAttackedBlock.equals(blockPos)) {
                        breakProgress = breakProgress + ((1.0 - breakProgress) * 0.5);
                    } else {
                        breakProgress = 0.0;
                    }
                } else {
                    breakProgress = 0.0;
                }

                // Determine block type value
                double blockValue = 0.0;
                if (distance <= normalReach) {
                    if (blockName.contains("diamond") || blockName.contains("gold")) {
                        blockValue = 1.0;
                    } else if (blockName.contains("redstone") || blockName.contains("iron")) {
                        blockValue = 0.9;
                    } else if (blockPos.getY() >= MIN_Y_LEVEL && blockPos.getY() <= MAX_Y_LEVEL) {
                        if (blockName.contains("stone") && !blockName.contains("bedrock")) {
                            blockValue = 0.6;
                        }
                    }
                }

                // Reset break progress if block value is 0
                if (blockValue == 0.0) {
                    breakProgress = 0.0;
                }

                targetBlockInfo = Arrays.asList(blockValue, inverseDistance, breakProgress);
                state.put("target_block", targetBlockInfo);

            } else {
                targetBlockInfo = Arrays.asList(0.0, 0.0, 0.0);
                state.put("target_block", targetBlockInfo);
            }

            // Ensure broken_blocks always has exactly 1 entry
            List<Object> normalizedBrokenBlocks;
            if (brokenBlocks.isEmpty()) {
                normalizedBrokenBlocks = Arrays.asList(0.0, 0.0, 0.0, 0.0);  // Use 0.0 instead of 0
            } else {
                // Take most recent block break if multiple exist
                Map<String, Object> lastBrokenBlock = brokenBlocks.get(brokenBlocks.size() - 1);
                
                // Convert values to double explicitly
                Number blockTypeNum = (Number)lastBrokenBlock.get("blocktype");
                double blockType = blockTypeNum.doubleValue();
                double blockX = 0.0;
                double blockY = 0.0;
                double blockZ = 0.0;
                double newRevealedBlocks = 0.0; // Initialize with default
                
                // Check if new_revealed_blocks is present
                if (lastBrokenBlock.containsKey("new_revealed_blocks")) {
                    Number revealedNum = (Number)lastBrokenBlock.get("new_revealed_blocks");
                    newRevealedBlocks = revealedNum.doubleValue();
                }

                // Check target block position relative to player if there is one 
                if (hitResult != null && hitResult.getType() == HitResult.Type.BLOCK) {
                    BlockHitResult blockHit = (BlockHitResult) hitResult;
                    BlockPos blockPos = blockHit.getBlockPos();
                    float currentYaw = player.getYaw();
                    
                }

                normalizedBrokenBlocks = Arrays.asList(blockType, blockX, blockY, newRevealedBlocks);
            }
            state.put("broken_blocks", normalizedBrokenBlocks);

            state.put("surrounding_blocks", getSurroundingBlocks(player));

            if (DEBUG_MODE) {
                Map<String, Object> logState = new HashMap<>(state);
                logState.remove("surrounding_blocks");
                System.out.println("[State] " + gson.toJson(logState));
            }
        }

        return state;
    }

    private List<List<Double>> getSurroundingBlocks(ClientPlayerEntity player) {
        List<List<Double>> grid = new ArrayList<>();
        BlockPos playerPos = player.getBlockPos();
        int normalBlockCount = 0;
        int oreCount = 0;
        int totalBlocks = 0;

        for (int dy = -OBSERVATION_RADIUS_Y_DOWN; dy <= OBSERVATION_RADIUS_Y_UP; dy++) {
            List<Double> row = new ArrayList<>();
            for (int dz = -OBSERVATION_RADIUS_Z; dz <= OBSERVATION_RADIUS_Z; dz++) {
                for (int dx = -OBSERVATION_RADIUS_X; dx <= OBSERVATION_RADIUS_X; dx++) {
                    BlockPos pos = playerPos.add(dx, dy, dz);
                    BlockState blockState = client.world.getBlockState(pos);
                    String blockName = blockState.getBlock().getTranslationKey().toLowerCase();
                    double value = 0.5; // Default value for most blocks

                    // Check for regular stone-type blocks (0.6)
                    if (blockName.contains("stone") || 
                        blockName.contains("granite") || 
                        blockName.contains("diorite") || 
                        blockName.contains("andesite") || 
                        blockName.contains("cobblestone") ||
                        blockName.contains("deepslate")) {
                        value = 0.6;
                        normalBlockCount++;
                    }
                    // Check for valuable ores (0.8-1.0)
                    else if (blockName.contains("diamond") || blockName.contains("gold")) {
                        value = 1.0;
                        oreCount++;
                    }
                    else if (blockName.contains("iron") || blockName.contains("redstone")) {
                        value = 0.9;
                        oreCount++;
                    }
                    else if (blockName.contains("coal") || 
                            blockName.contains("copper") || 
                            blockName.contains("emerald") || 
                            blockName.contains("lapis")) {
                        value = 0.8;
                        oreCount++;
                    }
                    // Check for dangerous blocks (0.0)
                    else if (blockName.contains("lava") || 
                            blockName.contains("fire") || 
                            blockName.contains("magma") || 
                            blockState.getBlock() instanceof Monster) {
                        value = 0.0;
                    }

                    totalBlocks++;
                    row.add(value);
                }
            }
            grid.add(row);
        }

        if(DEBUG_MODE) {
            System.out.println("[Blocks] Normal blocks: " + normalBlockCount + 
                            ", Ores: " + oreCount + 
                            "/" + totalBlocks + " total blocks in observation");
        }

        return grid;
    }
    
    /**
     * Sends the current player state over the WebSocket connection.
     *
     * @param conn The WebSocket connection.
     */
    private void sendPlayerState(WebSocket conn) {
        if (!isConnected || conn == null) {
            return;
        }
        try {
            Map<String, Object> state = getPlayerState();
            String stateJson = gson.toJson(state);
            conn.send(stateJson);
            brokenBlocks.clear();
            
            if (DEBUG_MODE) {
                long endTime = System.currentTimeMillis();
                System.out.println("[State] State sent at: " + endTime);
            }

            
        } catch (Exception e) {
            System.out.println("[WebSocket] Failed to send state - connection may be closed");
            if (DEBUG_MODE) {
                e.printStackTrace();
            }
            // Do not call cleanup here
        }
    }

    /**
     * Schedule state sending with debouncing.
     *
     * @param conn The WebSocket connection.
     */
    private synchronized void scheduleStateSend(WebSocket conn) {
        final long actionStartTime = System.currentTimeMillis();
        
        if (stateSendTask != null && !stateSendTask.isDone()) {
            stateSendTask.cancel(false);
        }
        
        stateSendTask = scheduler.schedule(() -> {
            sendPlayerState(conn);
            if (DEBUG_MODE) {
                long totalDuration = System.currentTimeMillis() - actionStartTime;
                System.out.println("[Timing] Total action cycle time: " + totalDuration + "ms");
            }
        }, STATE_SEND_DELAY, TimeUnit.MILLISECONDS);
    }

    /**
     * Executes a key action with a specified duration and schedules state sending.
     *
     * @param key        The key binding to execute.
     * @param durationMs Duration in milliseconds to hold the key.
     * @param conn       The WebSocket connection.
     */
    private void executeKeyAction(KeyBinding key, long durationMs, WebSocket conn) {
        // Cancel any existing release task for this key
        ScheduledFuture<?> existingTask = keyReleaseTasks.get(key);
        if (existingTask != null && !existingTask.isDone()) {
            existingTask.cancel(false);
        }

        key.setPressed(true);

        // Schedule key release
        ScheduledFuture<?> releaseTask = scheduler.schedule(() -> {
            key.setPressed(false);
            KeyBinding.updatePressedStates();
            keyReleaseTasks.remove(key);
        }, durationMs, TimeUnit.MILLISECONDS);

        keyReleaseTasks.put(key, releaseTask);

        // Schedule state send
        scheduleStateSend(conn);
    }

    /**
     * Executes the attack action, keeping the attack key pressed until the next action.
     *
     * @param key  The key binding for attack.
     * @param conn The WebSocket connection.
     */
    private void executeAttackAction(KeyBinding key, WebSocket conn) {
        if (DEBUG_MODE) {
            System.out.println("[Attack] Starting continuous attack action for client on port: " + PORT);
        }

        // Get currently targeted block
        var hitResult = client.player.raycast(20.0, 0.0f, false);
        if (hitResult != null && hitResult.getType() == HitResult.Type.BLOCK) {
            BlockHitResult blockHit = (BlockHitResult) hitResult;
            BlockPos newTargetBlock = blockHit.getBlockPos();
            
            // Reset progress if attacking a different block
            if (lastAttackedBlock == null || !lastAttackedBlock.equals(newTargetBlock)) {
                breakProgress = 0.0;
            }
            
            lastAttackedBlock = newTargetBlock;
            otherActionsSinceAttack = 0;
        }

        key.setPressed(true);
        KeyBinding.updatePressedStates();

        scheduleStateSend(conn);
    }

    /**
     * Executes a combined action: jumping and then walking forward, and schedules state sending.
     *
     * @param jumpKey    The key binding for jumping.
     * @param forwardKey The key binding for walking forward.
     * @param conn       The WebSocket connection.
     */
    private void executeJumpWalkForward(KeyBinding jumpKey, KeyBinding forwardKey, WebSocket conn) {
        // Cancel any existing tasks for these keys except attack and sneak
        cancelOngoingActionsExcept(Arrays.asList(jumpKey, forwardKey, client.options.sneakKey, client.options.attackKey));

        jumpKey.setPressed(true);

        // Schedule jump key release after 10ms
        scheduler.schedule(() -> {
            jumpKey.setPressed(false);
            KeyBinding.updatePressedStates();

            // Press forward key
            executeKeyAction(forwardKey, EXTENDED_ACTION_TIME, conn);
        }, 10, TimeUnit.MILLISECONDS);
    }

    /**
     * Executes a no-op action and schedules state sending.
     *
     * @param conn The WebSocket connection.
     */
    private void executeNoOpAction(WebSocket conn) {
        // No operation, just send the current state back
        scheduleStateSend(conn);
    }

    // Add the following method to define and set the inventory preset


    /**
     * Applies Night Vision effect to the player for 5 minutes.
     *
     * @param player The player entity.
     */
    private void applyNightVision(PlayerEntity player) {
        StatusEffectInstance nightVision = new StatusEffectInstance(StatusEffects.NIGHT_VISION, 6000, 0, false, false);
        player.addStatusEffect(nightVision);
        if (DEBUG_MODE) {
            System.out.println("[NightVision] Applied Night Vision effect for 5 minutes.");
        }
    }

    private void respawnPlayer() {
        if (client.player != null && !client.player.isAlive()) {
            // Schedule respawn on the main client thread
            client.execute(() -> {
                client.player.requestRespawn();
                if (DEBUG_MODE) {
                    System.out.println("[Reset] Player respawned");
                }
            });
        }
    }



    /**
     * Calculates the number of new blocks revealed adjacent to the broken block.
     *
     * @param world The game world.
     * @param pos   The position of the broken block.
     * @return The count of new blocks revealed.
     */
    private int calculateNewRevealedBlocks(World world, BlockPos pos) {
        if (world == null) {
            if (DEBUG_MODE) {
                System.out.println("[calculateNewRevealedBlocks] World is null.");
            }
            return 0;
        }
        
        int count = 0;
        for (Direction direction : Direction.values()) {
            BlockPos adjacentPos = pos.offset(direction);
            BlockState adjacentState = world.getBlockState(adjacentPos);
            Block adjacentBlock = adjacentState.getBlock();
            
            // Skip if adjacent block is air
            if (adjacentState.isAir()) {
                if (DEBUG_MODE) {
                    System.out.println("[calculateNewRevealedBlocks] Adjacent block at " + adjacentPos + " is air.");
                }
                continue;
            }

            // Count covered sides for this adjacent block
            int coveredSides = 0;
            int openSides = 0;
            Direction openSide = null;
            
            for (Direction dir : Direction.values()) {
                BlockPos checkPos = adjacentPos.offset(dir);
                if (!world.getBlockState(checkPos).isAir()) {
                    coveredSides++;
                } else {
                    openSides++;
                    openSide = dir;
                }
            }

            // Special handling for bedrock
            boolean isBedrock = adjacentBlock.getTranslationKey().toLowerCase().contains("bedrock");
            
            // Block is newly revealed if:
            // - Regular blocks: has exactly 5 covered sides
            // - Bedrock: has exactly 2 open sides
            if (coveredSides == 5 || (isBedrock && openSides == 4)) {
                count++;
                
                if (count >= 5) break; // Max count is 5
            } else {
                if (DEBUG_MODE) {
                    System.out.println("[calculateNewRevealedBlocks] Block at " + adjacentPos + 
                        " has " + coveredSides + " covered sides, " + openSides + " open sides.");
                }
            }
        }
        
        if (DEBUG_MODE) {
            System.out.println("[calculateNewRevealedBlocks] Total new revealed blocks: " + count);
        }
        return count;
    }

    // Add new helper method:
    private void printFormattedState(Map<String, Object> state, long elapsedTime) {
        StringBuilder sb = new StringBuilder();
        sb.append("\r"); // Carriage return to overwrite previous output
        
        // Process surrounding blocks
        @SuppressWarnings("unchecked")
        List<List<Double>> surroundingBlocks = (List<List<Double>>) state.get("surrounding_blocks");
        Map<String, Integer> blockCounts = new HashMap<>();
        blockCounts.put("stone", 0);  // 0.6
        blockCounts.put("diamonds", 0); // 1.0
        blockCounts.put("gold", 0);    // 1.0
        blockCounts.put("iron", 0);    // 0.9
        blockCounts.put("other", 0);   // 0.8 or less
        int totalBlocks = 0;
        
        for (List<Double> row : surroundingBlocks) {
            for (Double value : row) {
                totalBlocks++;
                if (value == 0.6) blockCounts.put("stone", blockCounts.get("stone") + 1);
                else if (value == 1.0) blockCounts.put("diamonds", blockCounts.get("diamonds") + 1);
                else if (value == 0.9) blockCounts.put("iron", blockCounts.get("iron") + 1);
                else if (value == 0.8) blockCounts.put("gold", blockCounts.get("gold") + 1);
                else blockCounts.put("other", blockCounts.get("other") + 1);
            }
        }
        
        sb.append(String.format("Blocks: %d stone - %d diamonds - %d iron - %d gold - %d other - %d total\n",
            blockCounts.get("stone"), 
            blockCounts.get("diamonds"),
            blockCounts.get("iron"),
            blockCounts.get("gold"),
            blockCounts.get("other"),
            totalBlocks));

        // Position
        sb.append(String.format("XYZ: (%.2f, %.2f, %.2f)\n", 
            state.get("x"), state.get("y"), state.get("z")));

        // Rotation
        sb.append(String.format("YawPitch: (%.2f, %.2f)\n",
            state.get("yaw"), state.get("pitch")));

        // Status
        sb.append(String.format("Health Light: (%.1f, %.1f)\n",
            state.get("health"), state.get("light_level")));

        // Target block
        @SuppressWarnings("unchecked")
        List<Double> target = (List<Double>) state.get("target_block");
        sb.append(String.format("Target: [%.1f, %.1f]\n",
            target.get(0), target.get(1)));

        // Held item
        @SuppressWarnings("unchecked")
        List<Integer> heldItem = (List<Integer>) state.get("held_item");
        sb.append(String.format("Hand: %d\n", heldItem.get(0)));

        // Broken blocks
        @SuppressWarnings("unchecked")
        List<Object> brokenBlocks = (List<Object>) state.get("broken_blocks");
        sb.append(String.format("Broken: [%.1f, %.1f, %.1f, %.1f]\n",
            ((Number)brokenBlocks.get(0)).doubleValue(),
            ((Number)brokenBlocks.get(1)).doubleValue(),
            ((Number)brokenBlocks.get(2)).doubleValue(),
            ((Number)brokenBlocks.get(3)).doubleValue()));

        // Timing
        sb.append(String.format("TIME: %dms", elapsedTime));

        // Print the entire status
        System.out.print(sb.toString());
    }

}